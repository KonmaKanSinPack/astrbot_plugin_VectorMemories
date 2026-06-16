import json
import os
from dataclasses import dataclass
from datetime import datetime, time, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import AstrBotConfig, logger
from astrbot.api.event import AstrMessageEvent, MessageChain, MessageEventResult, filter
from astrbot.api.provider import ProviderRequest
from astrbot.api.star import Context, Star, register
from astrbot.core.agent.message import (
    AssistantMessageSegment,
    TextPart,
    UserMessageSegment,
)
from astrbot.core.utils.astrbot_path import get_astrbot_data_path
from openai import AsyncOpenAI

import json_repair

try:
    from .embedding_service import EmbeddingService
except ImportError:
    from embedding_service import EmbeddingService  # fallback for some loaders
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _default_state() -> Dict[str, Any]:
    now = _utc_now()
    return {
        "core_memory": [],
        "long_term": [],
        "medium_term": [],
        # "metadata": {
        #     "version": 1,
        #     "created_at": now,
        #     "last_update": now,
        #     "summary": {},
        # },
    }


class MemoryStore:
    """记忆文件的 JSON 持久化读写。

    Args:
        path: 记忆 JSON 文件的绝对路径。
    """

    def __init__(self, path: str):
        self.path = Path(path)

    def load(self) -> Dict[str, Any]:
        """读取记忆文件。文件不存在或损坏时返回默认空结构并自动创建文件。

        Returns:
            记忆状态字典，包含 core_memory / long_term / medium_term 三个列表。
        """
        if not self.path.exists():
            state = _default_state()
            self.save(state)
            return state
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("读取记忆文件失败，将使用默认结构: %s", exc)
            state = _default_state()
            self.save(state)
            return state

    def save(self, state: Dict[str, Any]) -> None:
        """将记忆状态写入 JSON 文件，自动创建父目录。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

class UserRoster:
    """管理 user_name → subject_id 的映射关系，持久化到 user_roster.json。

    用于在群聊中将用户昵称映射到其专用 subject_id，确保记忆检索能
    正确关联到具体的用户。
    """

    def __init__(self: str):
        path = os.path.join(get_astrbot_data_path(), "user_roster.json")
        self.path = Path(path)
        self.id_dict = self.load()

    def load(self) -> Dict[str, Any]:
        """读取映射文件。文件不存在或损坏时返回空字典。"""
        if not self.path.exists():
            state = {}
            self.save(state)
            return state
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.error("读取UserRoster文件失败，将使用默认结构: %s", exc)
            state = {}
            self.save(state)
            return state

    def save(self, state: Dict[str, Any]) -> None:
        """写入映射文件。"""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def update(self, k, v=None, delete=False):
        """更新或删除一条 user_name → subject_id 映射。

        Args:
            k: user_name。
            v: subject_id，delete=True 时可省略。
            delete: 为 True 时删除该映射，否则添加或更新。
        """
        if not delete:
            self.id_dict[k] = v
            self.save(self.id_dict)
        else:
            if k in self.id_dict:
                del self.id_dict[k]
                self.save(self.id_dict)
        logger.info(f"当前字典：{self.id_dict}")

    def check(self):
        """返回当前所有映射。"""
        return self.id_dict

@dataclass
class UpsertResult:
    added: int = 0
    updated: int = 0
    deleted: int = 0


@register("vector_memories", "兔子", "为大模型提供向量化记忆提示词", "1.4.0")
class SimpleMemoryPlugin(Star):
    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context
        self.config = config
        self.use_global = self.config.get("use_global", True)
        self.last_update: Dict[str, str] = {}
        self.user_roster = UserRoster()

        self.enable_vector = self.config.get("enable_vector_memory", False)
        self.retrieval_top_k = self.config.get("retrieval_top_k", 5)

        provider_source = self.config.get("embedding_provider_source", "astrbot")

        self.embedding_service = EmbeddingService(
            provider_source=provider_source,
            context=self.context if provider_source == "astrbot" else None,
            provider_id=self.config.get("embedding_provider_id", ""),
            api_base_url=self.config.get(
                "embedding_api_base_url", "https://api.openai.com/v1"
            ),
            api_key=self.config.get("embedding_api_key", ""),
            model_name=self.config.get(
                "embedding_model_name", "text-embedding-ada-002"
            ),
            dimensions=self.config.get("embedding_dimensions", 1536),
        )

        # 向量模式开启但后端不可用时，自动降级为精确匹配
        if self.enable_vector and not self.embedding_service.is_ready:
            logger.warning(
                "Vector memory enabled but embedding backend is not ready. "
                "Falling back to exact-match retrieval."
            )
            self.enable_vector = False

        logger.info(
            "[VectorMemories] 初始化完成 | "
            f"向量模式={'开启' if self.enable_vector else '关闭'} | "
            f"后端={provider_source} | "
            f"模型={self.embedding_service.model_name} | "
            f"维度={self.embedding_service.dimensions} | "
            f"就绪={'是' if self.embedding_service.is_ready else '否'} | "
            f"top_k={self.retrieval_top_k} | "
            f"core_memory=始终全量注入"
        )

    def process_mem_info(
        self,
        mem_snapshot: Dict[str, Any],
        id_list: Optional[List[str]] = None,
    ) -> str:
        """将记忆快照转换为 LLM 提示词可用的字符串。

        Args:
            mem_snapshot: 记忆状态字典（core_memory / long_term / medium_term）。
            id_list: 允许的 subject_id 白名单。为 None 时不做过滤（向量检索
                     已预过滤的场景）；否则仅保留匹配条目并分 subject_id 组织。

        Returns:
            格式化后的记忆文本块，无匹配时返回 "No relevant memories found."。
        """
        if id_list is None:
            # id_list 为 None 表示调用方已预过滤（向量检索路径），直接拼接
            final_mem_info = []
            for mem_type in ["core_memory", "long_term", "medium_term"]:
                mem_entries = mem_snapshot.get(mem_type, [])
                if mem_entries:
                    entries_text = "\n".join(
                        f"- memory_id:{e.get('memory_id')}, {e.get('content')})"
                        for e in mem_entries
                        if e.get("content")
                    )
                    final_mem_info.append(f"{mem_type}:\n{entries_text}\n")
            if not final_mem_info:
                return "No relevant memories found."
            return (
                "<Relevant memories>\n"
                + "\n".join(final_mem_info)
                + "\n</Relevant memories>"
            )

        # 精确匹配路径：按 subject_id 过滤并分组
        final_mem_info = []
        for mem_type in ["core_memory", "long_term", "medium_term"]:
            filtered_entries = []
            if mem_type not in mem_snapshot:
                continue
            mem_entries = mem_snapshot.get(mem_type, [])
            id_mem = {id_: [] for id_ in id_list}
            for entry in mem_entries:
                if entry.get("subject_id") in id_list:
                    id_mem[entry.get("subject_id")].append(
                        f"- memory_id:{entry.get('memory_id')}, {entry.get('content')})"
                    )

            filtered_entries.extend(
                f"<subject_id: {id_}>\n" + "\n".join(entries) + "\n</subject_id>\n"
                for id_, entries in id_mem.items()
                if entries
            )
            final_mem_info.append(
                f"{mem_type}:\n" + "\n".join(filtered_entries) + "\n"
            )

        if not final_mem_info:
            return "No relevant memories found."
        else:
            return (
                "<Relevant memories>\n"
                + "\n".join(final_mem_info)
                + "\n</Relevant memories>"
            )
    

    @filter.on_llm_request()
    async def add_mem_prompt(self, event: AstrMessageEvent, req: ProviderRequest, *_, **__):
        """LLM 请求钩子：在 system_prompt 中注入与当前消息最相关的记忆。

        向量模式启用时按语义相似度检索 top-k 记忆（先 subject_id 预过滤再排序）；
        否则按 subject_id 精确匹配并返回全部记忆（原有行为）。
        core_memory 默认始终全部包含，除非同时开启向量模式且配置了 include_all_core=false。
        """
        uid = event.unified_msg_origin
        subject_id = uid.split(":")[-1]
        msg_type = uid.split(":")[-2]
        sender_name = event.get_sender_name()

        # 构建 subject_id 白名单：群聊时包含群 ID 和 sender 的个人 ID
        if msg_type == "GroupMessage":
            id_list = [
                "global",
                subject_id,
                self.user_roster.id_dict.get(sender_name, ""),
            ]
        else:
            id_list = ["global", subject_id]
            if (
                sender_name not in self.user_roster.id_dict
                and msg_type != "GroupMessage"
            ):
                self.user_roster.update(sender_name, subject_id)

        mem_file_path = (
            os.path.join(get_astrbot_data_path(), "memory_store_global.json")
            if self.use_global
            else os.path.join(get_astrbot_data_path(), f"memory_store_{uid}.json")
        )
        state = MemoryStore(mem_file_path).load()

        # ---- core_memory：始终全部注入（数量少，是 AI 人格，不应被过滤） ----
        core_mem = state.get("core_memory", [])
        core_mem_list = []
        for entry in core_mem:
            if entry.get("content"):
                core_mem_list.append(
                    f"- memory_id:{entry.get('memory_id')}, "
                    f"{entry.get('content')}, "
                    f"subject_id: {entry.get('subject_id')}"
                )
        core_mem_info = "\n".join(core_mem_list)
        state.pop("core_memory", None)

        # ---- long_term / medium_term：向量相似度排序或精确匹配 ----
        if self.enable_vector and self.embedding_service.is_ready:
            query_text = event.message_str or ""
            query_emb = None
            if query_text:
                query_emb = await self.embedding_service.get_embedding(query_text)
                if query_emb:
                    preview = ", ".join(f"{v:.4f}" for v in query_emb[:5])
                    logger.info(
                        f"[VectorMemories] 查询向量已生成 | "
                        f"维度={len(query_emb)} | "
                        f"前5值=[{preview}...] | "
                        f"查询文本={query_text[:50]}..."
                    )
                else:
                    logger.warning("[VectorMemories] 查询向量生成失败，降级为精确匹配")

            if query_emb is not None:
                # 向量路径：先 subject_id 预过滤（隐私边界），再余弦相似度排序
                filtered_snapshot: Dict[str, List[Dict]] = {}
                for mem_type in ["long_term", "medium_term"]:
                    entries = state.get(mem_type, [])
                    candidates = [
                        e for e in entries if e.get("subject_id") in id_list
                    ]
                    if not candidates:
                        filtered_snapshot[mem_type] = []
                        continue
                    ranked = self.embedding_service.rank_memories(
                        query_emb, candidates, self.retrieval_top_k
                    )
                    filtered_snapshot[mem_type] = [mem for mem, _ in ranked]

                memory_snapshot = self.process_mem_info(filtered_snapshot)
            else:
                # embedding 失败 → 降级为精确匹配
                memory_snapshot = self.process_mem_info(state, id_list=id_list)
        else:
            # 向量模式关闭 → 精确匹配
            memory_snapshot = self.process_mem_info(state, id_list=id_list)

        ori_system_prompt = req.system_prompt or ""

        current_user_id = (
            subject_id
            if msg_type != "GroupMessage"
            else self.user_roster.id_dict.get(sender_name, "unknown")
        )
        current_group_id = (
            subject_id if msg_type == "GroupMessage" else "None (Private Chat)"
        )

        mem_prompt = (
            "\n\n====================\n"
            "### [CURRENT CHAT CONTEXT] ###\n"
            f"- 当前正在对你说话的用户名字 (Sender Name): {sender_name}\n"
            f"- 当前用户的专属 ID (User ID): {current_user_id}\n"
            f"- 当前所在的群组 ID (Group ID): {current_group_id}\n\n"
            "### [MEMORY SYSTEM RULES - STRICT] ###\n"
            f"1. 极其重要：除了 global 记忆外，你只能将带有 <subject_id: {current_user_id}> 或 <subject_id: {current_group_id}> 的记忆应用到当前用户身上！\n"
            f"2. 绝对禁止将其他用户的记忆（如提到其他 subject_id 的内容）当作当前用户 ({sender_name}) 的经历！如果记忆里的 subject_id 与当前 User ID 不匹配，说明那是别人的事，请保持客观，不要张冠李戴。\n"
            "3. core_memory 只表示 AI 自身人格、灵魂、价值观、思考方式、表达风格、稳定自我认知等高度抽象且长期稳定的内容。\n"
            "4. core_memory 绝对不是具体事实仓库，不应被理解为某个用户资料、某次对话经过、一次性事件、临时任务或外部世界的具体事实清单。\n"
            "5. 阅读和使用 core_memory 时，只能把它当作 AI 的人格底色与内在原则；如果内容是具体事实，应优先从 long_term 或 medium_term 理解。\n\n"
            "### [RETRIEVED MEMORIES] ###\n"
            f"<core_memory>\n{core_mem_info}\n</core_memory>\n"
            f"{memory_snapshot}\n"
            "====================\n"
        )

        req.system_prompt = ori_system_prompt + f"\n{mem_prompt}"
        logger.info(f"当前的系统提示词_SimpleMemory:{req.system_prompt}")

    @filter.command_group("mem")
    def mem(self, t):
        pass
    
    @mem.command("check")
    async def check(self, event: AstrMessageEvent):
        """查看上次 /mem gen 返回的原始更新内容。"""
        uid = event.unified_msg_origin
        if self.last_update.get(uid) is None:
            await self.context.send_message(
                uid, MessageChain().message("尚未进行过记忆更新。")
            )
        else:
            await self.context.send_message(
                uid,
                MessageChain().message(f"上次更新内容:\n{self.last_update[uid]}"),
            )
        event.stop_event()
    
    
    @mem.command("gen")
    async def gen(
        self, event: AstrMessageEvent, extra_prompt: str = "", use_full: str = ""
    ):
        """将对话历史和记忆快照发给 LLM，生成更新 JSON 并自动应用。

        Args:
            extra_prompt: 追加到提示词末尾的额外指令。
            use_full: "--full" 时使用全部对话历史。
        """
        mem_result = await self.send_prompt(
            event,
            extra_prompt=extra_prompt,
            full=(str(use_full).strip() == "--full"),
        )
        self.last_update[event.unified_msg_origin] = mem_result

        handle_result = await self._handle_apply(event, mem_result)
        logger.info(f"应用记忆结果:{handle_result}")
        message_chain = MessageChain().message(handle_result)
        await self.context.send_message(event.unified_msg_origin, message_chain)
        event.stop_event()

    @mem.command("test")

    async def test_embedding(self, event: AstrMessageEvent):
        """验证 embedding provider 是否可用。

        依次检查：
        1. 能否从 AstrBot Context 获取到 embedding provider
        2. 调用 get_embedding("Hello world") 看能否正常返回向量
        """
        uid = event.unified_msg_origin

        # Step 1: 检查 provider 是否存在
        get_all = getattr(self.context, "get_all_embedding_providers", None)
        if get_all is None:
            yield event.plain_result("[step 1] context 没有 get_all_embedding_providers 方法")
            return

        providers = get_all()
        if not providers:
            yield event.plain_result("[step 1] get_all_embedding_providers() 返回空列表，请在 WebUI 服务提供商页面配置 Embedding")
            return

        provider = providers[0]
        provider_type = type(provider).__name__
        yield event.plain_result(f"[step 1] 获取到 provider: {provider_type}")
        yield event.plain_result(f"查看config:{provider.provider_config}")

        # Step 2: 测试 get_embedding
        try:
            result = await provider.get_embedding("Hello world")
            if result is None:
                yield event.plain_result("[step 2] get_embedding 返回 None")
                return
            vec = list(result)
            preview = ", ".join(f"{v:.4f}" for v in vec[:5])
            yield event.plain_result(
                f"[step 2] embedding 调用成功 | 维度={len(vec)} | 前5值=[{preview}...]"
            )
        except Exception as e:
            yield event.plain_result(f"[step 2] get_embedding 失败: {e}")
            return

        yield event.plain_result("[结论] embedding provider 正常工作，可以开启向量模式")
        event.stop_event()

    @mem.command("rebuild")
    async def mem_rebuild(self, event):
        """重构记忆：备份当前文件 → 清空 → LLM 基于旧记忆从零重建。"""
        uid = event.unified_msg_origin
        mem_path = (
            os.path.join(get_astrbot_data_path(), f"memory_store_{uid}.json")
            if not self.use_global
            else os.path.join(get_astrbot_data_path(), "memory_store_global.json")
        )
        pre_mem_path = (
            os.path.join(get_astrbot_data_path(), f"memory_store_{uid}_pre.json")
            if not self.use_global
            else os.path.join(
                get_astrbot_data_path(), "memory_store_global_pre.json"
            )
        )
        # 优先使用已有的 _pre 备份，否则读取当前文件
        if os.path.exists(pre_mem_path):
            state_pre = MemoryStore(pre_mem_path).load()
        else:
            state_pre = MemoryStore(mem_path).load()

        # 去掉向量和元数据，只保留 LLM 需要的字段，避免撑爆 context
        clean_state = {}
        for mem_type in ["core_memory", "long_term", "medium_term"]:
            clean_state[mem_type] = []
            for entry in state_pre.get(mem_type, []):
                clean_state[mem_type].append({
                    "memory_id": entry.get("memory_id"),
                    "content": entry.get("content"),
                })

        old_mem_json = json.dumps(clean_state, ensure_ascii=False, indent=2)
        # 截断，防止超出模型 context 限制
        if len(old_mem_json) > 8000:
            old_mem_json = old_mem_json[:8000] + "\n... (truncated)"

        # 将当前文件重命名为备份 → 删除原文件，gen 会创建新的记忆文件
        try:
            os.rename(mem_path, pre_mem_path)
            os.remove(mem_path)
        except Exception as e:
            logger.info(f"发生错误:{e}")

        await self.gen(
            event,
            extra_prompt="这是你之前的记忆，根据这些记忆重构现在的记忆:\n"
            + old_mem_json,
        )
        event.stop_event()

    @mem.command("help")
    async def help(self, event: AstrMessageEvent):
        """显示 /mem 子命令的使用说明。"""
        yield event.plain_result(self._usage_manual())
        return
    
    @filter.llm_tool(name="update_user_roster_id_dict")
    async def update_user_roster_id_dict(
        self,
        event: AstrMessageEvent,
        user_name: Optional[str] = None,
        subject_id: Optional[str] = None,
        delete: bool = False,
    ) -> MessageEventResult:
        """更新或删除 user_name → subject_id 映射。

        当发现某记忆的主体与当前 user_name 不匹配但实际上是同一人时调用此工具。
        删除时传 user_name + delete=True，subject_id 可选。

        Args:
            user_name (str): 用户名字。
            subject_id (str): 该用户关联的 subject_id。
            delete (bool): True 时删除该映射，默认 False（添加/更新）。
        """
        if user_name is None:
            return "必须提供 user_name 参数。"
        if subject_id is None and not delete:
            return "必须提供 subject_id 参数。"

        self.user_roster.update(user_name, subject_id, delete=delete)
        return f"已更新 user_name '{user_name}' 与 subject_id '{subject_id}' 的映射关系。"

    @filter.llm_tool(name="search_memory_by_user_name")
    async def search_memory_by_user_name(
        self,
        event: AstrMessageEvent,
        user_name: Optional[str] = None,
        query: Optional[str] = "",
    ) -> MessageEventResult:
        """按 user_name 搜索该用户的记忆。

        向量模式启用且提供 query 时，按语义相似度返回 top-k；
        否则返回该 subject_id 下的全部记忆。
        

        Args:
            user_name (str): 用户名字。
            query (str): 可选搜索文本，用于语义相似度匹配。
        """
        if user_name is None:
            return "必须提供 user_name 参数。"

        subject_id = self.user_roster.id_dict.get(user_name)
        if subject_id is None:
            return (
                f"未找到与 user_name '{user_name}' 相关的 subject_id。"
                f"这是当前的 user_name-subject_id 映射: {self.user_roster.id_dict}。"
                "你可以根据这个内容查看是否有实际上是同一人但名字不同的情况。"
                "如果有，你必须调用update_user_roster_id_dict来把当前的user_name更新映射列表"
            )

        mem_file_path = (
            os.path.join(
                get_astrbot_data_path(),
                f"memory_store_{event.unified_msg_origin}.json",
            )
            if not self.use_global
            else os.path.join(get_astrbot_data_path(), "memory_store_global.json")
        )
        state = MemoryStore(mem_file_path).load()

        # 向量语义搜索（仅在 query 参数提供且向量模式可用时生效）
        if query and self.enable_vector and self.embedding_service.is_ready:
            query_emb = await self.embedding_service.get_embedding(query)
            if query_emb:
                all_ranked: List[Tuple[Dict, float]] = []
                for mem_type in ["core_memory", "long_term", "medium_term"]:
                    candidates = [
                        e
                        for e in state.get(mem_type, [])
                        if e.get("subject_id") == subject_id
                    ]
                    if candidates:
                        ranked = self.embedding_service.rank_memories(
                            query_emb, candidates, self.retrieval_top_k
                        )
                        for mem, score in ranked:
                            mem["_mem_type"] = mem_type
                        all_ranked.extend(ranked)

                all_ranked.sort(key=lambda x: x[1], reverse=True)
                top = all_ranked[: self.retrieval_top_k]
                if top:
                    result_lines = ["相关记忆 (按相似度排序):"]
                    for mem, score in top:
                        mem_type = mem.pop("_mem_type", "unknown")
                        result_lines.append(
                            f"- [{mem_type}] memory_id:{mem['memory_id']}: "
                            f"{mem['content']} (相似度: {score:.3f})"
                        )
                    return "\n".join(result_lines)
                return f"未找到与 '{user_name}' 相关的记忆。"

        # 降级：精确 subject_id 匹配
        mem_info = self.process_mem_info(state, id_list=[subject_id])
        return mem_info

    @filter.llm_tool(name="check_user_roster_id_dict")
    async def check_user_roster_id_dict(
        self, event: AstrMessageEvent
    ) -> MessageEventResult:
        """检查当前的 user_name → subject_id 映射关系。"""
        return self.user_roster.id_dict

    @filter.llm_tool(name="update_one_memory")
    async def update_one_memory(
        self,
        event: AstrMessageEvent,
        memory_type: Optional[str] = None,
        action_type: Optional[str] = None,
        memory_id: Optional[str] = None,
        content: Optional[str] = None,
        category: Optional[str] = None,
        importance: Optional[int] = None,
        expires_at: Optional[str] = None,
        subject_id: Optional[str] = None,
    ) -> MessageEventResult:
        """精准管理单条记忆（增/改/删）。

        规则：
        - core_memory 只能存 AI 人格/灵魂/价值观等抽象内容，禁止存具体事实。
        - subject_id 必须是具体用户 ID，禁止把用户私人信息存入 "global"。
        - 修改/删除前必须先用 search_memory_by_user_name 确认 memory_id。

        Args:
            memory_type (str): core_memory | long_term | medium_term。
            action_type (str): upsert | delete。
            memory_id (str): 唯一标识，新增时自生成，更新/删除时必填。
            content (str): 记忆文本，upsert 时必填。
            category (str): profile | preference | task | fact，默认 fact。
            importance (int): 1-5，默认 3。
            expires_at (str): YYYY-MM-DD，留空永久。
            subject_id (str): 归属者 ID，仅客观真理可用 "global"。
        """
        cur_state = {
            "memory_type": memory_type,
            "action_type": action_type,
            "memory_id": memory_id,
            "content": content,
            "category": category,
            "importance": importance,
            "expires_at": expires_at,
        }
        logger.info("update_one_memory called with: %s", cur_state)

        if memory_type not in {"core_memory", "long_term", "medium_term"}:
            return "无效的记忆类型，memory_type仅支持 core_memory、long_term 或 medium_term。"
        if action_type not in {"upsert", "delete"}:
            return "无效的操作类型，action_type仅支持 upsert 或 delete。"
        if not memory_id:
            return "必须提供 memory_id"
        if action_type == "upsert" and not content:
            return "upsert 操作必须提供 content。"

        if action_type == "upsert":
            operations = {
                memory_type: {
                    "upsert": [
                        {
                            "memory_id": memory_id,
                            "content": content,
                            "category": category or "fact",
                            "importance": (
                                importance if importance is not None else 3
                            ),
                            "expires_at": expires_at or "",
                            "subject_id": subject_id or "global",
                        }
                    ],
                    "delete": [],
                }
            }
        else:
            operations = {
                memory_type: {
                    "upsert": [],
                    "delete": [memory_id],
                }
            }

        mem_to_update = json.dumps(operations, ensure_ascii=False)
        report = await self._handle_apply(event, mem_to_update)
        logger.info("State update report: %s", report)
        if report.startswith("Update Failed"):
            return report
        else:
            return report

    @filter.llm_tool(name="delete_several_memories")
    async def delete_several_memories(
        self,
        event: AstrMessageEvent,
        memory_type: Optional[str] = None,
        memory_ids_to_delete_list: list = [],
    ) -> MessageEventResult:
        """批量删除同类型的多条记忆。

        删除前必须先调用 search_memory_by_user_name 确认 memory_id。

        Args:
            memory_type (str): core_memory | long_term | medium_term。
            memory_ids_to_delete_list (list): 待删除的 memory_id 列表。
        """
        if memory_type not in {"core_memory", "long_term", "medium_term"}:
            return (
                "无效的记忆类型，必须提供 memory_type参数，"
                "且memory_type仅支持 core_memory、long_term 或 medium_term。"
            )
        if not memory_ids_to_delete_list:
            return "必须提供 memory_ids_to_delete_list参数"
        if not isinstance(memory_ids_to_delete_list, list):
            return (
                "memory_ids_to_delete_list 必须是一个列表，"
                '格式示例: ["memory_id1", "memory_id2", ...]。'
            )

        reports = []
        for memory_id in memory_ids_to_delete_list:
            operations = {
                memory_type: {
                    "upsert": [],
                    "delete": [memory_id],
                }
            }
            mem_to_update = json.dumps(operations, ensure_ascii=False)
            reports.append(await self._handle_apply(event, mem_to_update))

        return "\n".join(reports)

    @mem.command("apply")
    async def apply(self, event: AstrMessageEvent):
        """手动应用 LLM 返回的 JSON 记忆更新。"""
        raw_message = (event.message_str or "").strip()
        subcommand, payload = self._parse_arguments(raw_message)

        result = await self._handle_apply(event, payload)
        yield event.plain_result(result)
        return

    def _parse_arguments(self, message: str) -> Tuple[str, str]:
        """解析 /mem apply 后的子命令和参数。

        Returns:
            (subcommand, payload) 元组，subcommand 为 "prompt"/"apply"/"help"。
        """
        normalized = message.lstrip("/").strip()
        if normalized.lower().startswith("memory"):
            normalized = normalized[6:].strip()

        if not normalized:
            return "help", ""

        parts = normalized.split(maxsplit=1)
        head = parts[0].lower()
        tail = parts[1].strip() if len(parts) > 1 else ""

        if head in {"prompt", "p"}:
            return "prompt", tail
        if head in {"apply", "a"}:
            return "apply", tail

        return "prompt", normalized

    def _usage_manual(self) -> str:
        """返回 /mem 子命令的使用说明。"""
        return (
            "记忆指令使用方式:\n"
            "1. /mem gen 生成给大模型使用的长中短期记忆。"
            "使用--full参数可使用全部对话历史。\n"
            "2. /mem check 查看上次记忆更新结果。\n"
            "建议流程: /mem gen -> 让大模型总结并应用记忆 -> /mem check 查看结果。"
        )

    async def send_prompt(self, event, extra_prompt="", full=False):
        """将记忆更新提示词发送给 LLM，返回其生成的 JSON 文本。

        Args:
            event: AstrMessageEvent。
            extra_prompt: 追加到提示词末尾的额外指令。
            full: True 时使用全部会话历史，False 时仅最近一轮。

        Returns:
            LLM 返回的完整文本（期望是 JSON 记忆操作）。
        """
        uid = event.unified_msg_origin

        conv_mgr = self.context.conversation_manager
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        conversation = await conv_mgr.get_conversation(uid, curr_cid)
        history = (
            json.loads(conversation.history)
            if conversation and conversation.history
            else []
        )

        person_prompt = await self.context.persona_manager.get_default_persona_v3(uid)
        if not person_prompt:
            person_prompt = self.context.provider_manager.selected_default_persona["prompt"]

        mem_prompt = self._handle_prompt(event, history, full)
        if extra_prompt != "":
            mem_prompt = extra_prompt + "\n" + mem_prompt
        logger.info(f"查看mem:{mem_prompt}")
        provider = self.context.get_using_provider()
        llm_resp = await provider.text_chat(
            prompt=mem_prompt,
            session_id=None,
            contexts=history,
            image_urls=[],
            func_tool=None,
            system_prompt=person_prompt,
        )
        return llm_resp.completion_text

    def _handle_prompt(self, event: AstrMessageEvent, history: str, full=False) -> str:
        """构建记忆更新提示词。

        拼接"任务指令 + 当前记忆快照 + 当前 subject_id + JSON 输出格式要求"，
        full=True 时要求基于全部对话更新，否则只基于最新一轮。

        Returns:
            完整提示词字符串。
        """
        uid = event.unified_msg_origin
        mem_file_path = (
            os.path.join(get_astrbot_data_path(), f"memory_store_{uid}.json")
            if not self.use_global
            else os.path.join(get_astrbot_data_path(), "memory_store_global.json")
        )
        # 记忆文件不存在或显式指定 --full 时使用全部历史
        if not Path(mem_file_path).exists() or full:
            task_prompt = (
                "please refresh core/long-term/medium-term memory based on "
                "the entire conversation.\n"
            )
        else:
            task_prompt = (
                "please refresh core/long-term/medium-term memory based on "
                "the latest conversation.\n"
            )
        state = MemoryStore(mem_file_path).load()
        state.pop("metadata", None)
        logger.info("创建记忆提示词，操作者: %s", uid)
        
        memory_snapshot = json.dumps(state, ensure_ascii=False, indent=2)
        cur_mem_prompt = (
            "You are an intelligent agent with a structured memory system. Below is your current memory snapshot.\n"
            "When updating your memories, follow these principles:\n"
            "1. Do NOT add any memory that already exists or is highly similar to an existing one.\n"
            "2. Proactively identify and forget memories that are outdated, irrelevant, or of low value.\n"
            "3. Keep your memory concise, focused, and up-to-date. Remove any redundant, obsolete, or trivial information.\n"
            "4. Only retain information that is useful for future reasoning, continuity, or identity.\n"
            "5. When in doubt, prefer fewer, higher-quality memories over more, lower-quality ones.\n"
            "6. Core memory must remain stable and only change when absolutely necessary.\n"
            "7. Core memory is reserved only for the AI's persona, soul-level self-concept, values, worldview, thinking patterns, and enduring inner principles.\n"
            "8. Do NOT store concrete facts in core memory, including user profile facts, specific conversation events, one-off experiences, temporary tasks, or ordinary world facts; those belong in long_term or medium_term.\n"
            "9. Any new content written into core_memory must strictly follow the rule above.\n"
            "10. short-term memory is not needed to generate. Make sure all memory you generate is either core, long-term, or medium-term.\n"
            "\n**[Current Memory Snapshot]**\n"
            f"{memory_snapshot}"
            "\n**[Current subject_id]，use it if this memory is associated with a specific user or group**\n"
            f"{uid}\n"
        )
        
        template = (
            task_prompt +
            cur_mem_prompt + 
            self.config.get("mem_prompt", "") +
            "output JSON with the following sections (each is required and serves a distinct purpose):\n"
            "- summary: concise highlights of any changes across memories.\n"
            "- core_memory: only the AI's enduring persona, soul-level self-concept, values, worldview, and thinking style; never concrete user facts, event records, or ordinary factual notes.\n"
            "- long_term: durable knowledge, goals, reusable user facts, and other concrete information worth keeping across many sessions; update cautiously.\n"
            "- medium_term: active themes, recent continuity, short-to-mid horizon tasks, and concrete contextual facts spanning recent sessions.\n"
            "Special rule: if a memory is concrete and factual, it must not go into core_memory even if it feels important.\n"
            "JSON Format:\n"
            "{\n"
            "  \"summary\": {\n"
            "    \"core_memory_highlights\": \"<summary of core memory changes>\",\n"
            "    \"long_term_highlights\": \"<summary of long-term changes>\",\n"
            "    \"medium_term_highlights\": \"<summary of medium-term changes>\",\n"
            "  },\n"
            "  \"core_memory\": {\n"
            "    \"upsert\": [{\n"
            "      \"memory_id\": \"reuse or system generated\",\n"
            "      \"content\": \"memory text\",\n"
            "      \"category\": \"profile|preference|task|fact\",\n"
            "      \"importance\": 1-5,\n"
            "      \"expires_at\": \"YYYY-MM-DD or leave blank\"\n"
            "      \"subject_id\": \"(who/which group this memory is associated with; use 'global' means global memory)\"\n"
            "    }],\n"
            "    \"delete\": [\"memory_id to delete\"]\n"
            "  },\n"
            "  \"long_term\": { same structure as core_memory },\n"
            "  \"medium_term\": { same structure as core_memory },\n"
            "}\n\n"
            "If no changes are needed, return empty upsert/delete and explain why in the summary."
        )
        # logger.info(f"记忆提示词内容:{template}")
        return template

    async def _handle_apply(self, event, payload_text: str) -> str:
        """解析 LLM 返回的 JSON 记忆操作，应用到内存状态并持久化。

        先尝试 json_repair 容错解析，失败后回退到标准 json.loads。
        向量模式启用时，在持久化之后为新增/修改的条目生成 embedding 并再次保存。

        Args:
            event: AstrMessageEvent。
            payload_text: LLM 输出的原始文本（可能包含 ```json 代码块）。

        Returns:
            操作结果报告字符串。
        """
        payload_text = payload_text.strip()
        if not payload_text:
            return "请提供大模型返回的 JSON 内容。"

        json_text = self._extract_json_block(payload_text)
        if json_text is None:
            return "未能解析 JSON，请直接粘贴模型输出或 ```json ``` 代码块。"

        # 容错解析 → 标准解析的降级链
        try:
            operations = json_repair.loads(json_text)
            logger.info("JSON parsed successfully: %s", operations)
        except Exception as e:
            logger.warning("JSON repair failed, fallback to standard parser: %s", e)
            try:
                operations = json.loads(json_text.strip())
            except json.JSONDecodeError as exc:
                return f"JSON parsing failed: {exc}"

        uid = event.unified_msg_origin
        mem_file_path = (
            os.path.join(get_astrbot_data_path(), f"memory_store_{uid}.json")
            if not self.use_global
            else os.path.join(get_astrbot_data_path(), "memory_store_global.json")
        )
        store = MemoryStore(mem_file_path)
        state = store.load()

        report = self._apply_operations(state, operations)
        store.save(state)

        # 向量模式：为缺少 embedding 的条目批量生成向量并二次保存
        if self.enable_vector and self.embedding_service.is_ready:
            await self._embed_new_entries(state)
            store.save(state)

        return report

    async def _embed_new_entries(self, state: Dict[str, Any]) -> None:
        """为 state 中缺少 embedding 的记忆条目批量生成向量并写回。"""
        texts_to_embed: List[Tuple[str, str, str]] = []
        for mem_type in ["core_memory", "long_term", "medium_term"]:
            for entry in state.get(mem_type, []):
                if entry.get("content") and not entry.get("embedding"):
                    texts_to_embed.append(
                        (mem_type, entry["memory_id"], entry["content"])
                    )

        if not texts_to_embed:
            return

        texts = [t[2] for t in texts_to_embed]
        embeddings = await self.embedding_service.get_embeddings(texts)

        emb_by_id: Dict[str, List[float]] = {}
        for (_, mem_id, _), emb in zip(texts_to_embed, embeddings):
            if emb is not None:
                emb_by_id[mem_id] = emb

        if not emb_by_id:
            return

        for mem_type in ["core_memory", "long_term", "medium_term"]:
            for entry in state.get(mem_type, []):
                if entry["memory_id"] in emb_by_id:
                    entry["embedding"] = emb_by_id[entry["memory_id"]]

        logger.info(
            f"Generated embeddings for {len(emb_by_id)}/{len(texts_to_embed)} entries"
        )

    def _extract_json_block(self, text: str) -> Optional[str]:
        """从 LLM 原始输出中提取 JSON 块。

        依次尝试：``` 代码块 → 直接 JSON 字符串 → 混合文本中的首个合法 JSON。

        Returns:
            提取到的纯 JSON 字符串，提取失败返回 None。
        """
        stripped = text.strip()
        if not stripped:
            return None
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3 and lines[-1].startswith("```"):
                return "\n".join(lines[1:-1]).strip()
            if stripped.startswith("```json"):
                return "\n".join(lines[1:-1]).strip()
            return None
        if stripped[0] in "[{" and stripped[-1] in "]}":
            return stripped

        # 从混合文本中恢复首个合法 JSON 对象/数组
        decoder = json.JSONDecoder()
        for i, ch in enumerate(stripped):
            if ch not in "[{":
                continue
            try:
                _, end = decoder.raw_decode(stripped[i:])
                return stripped[i : i + end]
            except Exception:
                continue
        return None

    def _apply_operations(
        self, state: Dict[str, Any], operations: Dict[str, Any]
    ) -> str:
        """将 LLM 返回的 JSON 操作应用到内存状态（core/long/medium 三种类型）。

        Returns:
            操作结果报告字符串。
        """
        now = _utc_now()
        report_lines: List[str] = []

        core_result = self._upsert_and_delete(
            state.setdefault("core_memory", []),
            operations.get("core_memory", {}),
            True,
            now,
        )
        lt_result = self._upsert_and_delete(
            state.setdefault("long_term", []),
            operations.get("long_term", {}),
            True,
            now,
        )
        mt_result = self._upsert_and_delete(
            state.setdefault("medium_term", []),
            operations.get("medium_term", {}),
            True,
            now,
        )

        state.pop("metadata", None)
        report_lines.append(self._format_report_line("核心记忆", core_result))
        report_lines.append(self._format_report_line("长期", lt_result))
        report_lines.append(self._format_report_line("中期", mt_result))

        summary_block = operations.get("summary")
        if isinstance(summary_block, dict) and summary_block:
            core_high = summary_block.get("core_memory_highlights", "无")
            lt_high = summary_block.get("long_term_highlights", "无")
            mt_high = summary_block.get("medium_term_highlights", "无")
            report_lines.append(
                "概述:\n- 核心: "
                + core_high
                + "\n- 长期: "
                + lt_high
                + "\n- 中期: "
                + mt_high
            )

        return "记忆已更新:\n" + "\n".join(report_lines)

    def _upsert_and_delete(
        self,
        bucket: List[Dict[str, Any]],
        operations: Dict[str, Any],
        is_long_term: bool,
        timestamp: str,
    ) -> UpsertResult:
        """对单个记忆桶执行 upsert/delete 操作。

        核心逻辑：
        - 用 memory_id 去重，新增的补 created_at，更新的保留原 created_at。
        - 更新时如果 content 发生变化，清除旧 embedding，由 _handle_apply 重新生成。
        - 删除时直接从 index 中移除。

        Returns:
            (added, updated, deleted) 统计。
        """
        result = UpsertResult()
        index = {
            item.get("memory_id"): item
            for item in bucket
            if item.get("memory_id")
        }

        upserts = operations.get("upsert") or []
        if not isinstance(upserts, list):
            upserts = []

        for raw_entry in upserts:
            if not isinstance(raw_entry, dict):
                continue
            content = (raw_entry.get("content") or "").strip()
            subject_id = (raw_entry.get("subject_id") or "global").strip()
            if not content:
                continue

            entry = raw_entry.copy()
            entry["content"] = content
            entry["subject_id"] = subject_id
            entry["updated_at"] = timestamp
            entry.setdefault("category", "fact" if is_long_term else "task")
            entry.setdefault("importance", 3)

            entry_id = entry.get("memory_id") or self._generate_entry_id(
                is_long_term
            )
            entry["memory_id"] = entry_id

            if entry_id in index:
                old_entry = index[entry_id]
                entry.setdefault(
                    "created_at", old_entry.get("created_at", timestamp)
                )
                # 内容变更时清除旧 embedding，后续由 _handle_apply 重新生成
                if old_entry.get("content") != entry["content"]:
                    entry.pop("embedding", None)
                index[entry_id].update(entry)
                result.updated += 1
            else:
                entry.setdefault("created_at", timestamp)
                index[entry_id] = entry
                result.added += 1

        deletes = operations.get("delete") or []
        if not isinstance(deletes, list):
            deletes = []

        for entry_id in deletes:
            if entry_id in index and entry_id:
                del index[entry_id]
                result.deleted += 1

        bucket.clear()
        bucket.extend(index.values())
        return result

    async def get_all_conversation(self, event: AstrMessageEvent) -> str:
        """获取当前会话的完整对话历史（JSON 字符串）。"""
        uid = event.unified_msg_origin
        conv_mgr = self.context.conversation_manager
        curr_cid = await conv_mgr.get_curr_conversation_id(uid)
        conversation = await conv_mgr.get_conversation(uid, curr_cid)
        return conversation.history

    def _generate_entry_id(self, is_long_term: bool) -> str:
        """生成唯一 memory_id：lt-{timestamp} 或 st-{timestamp}。"""
        prefix = "lt" if is_long_term else "st"
        return f"{prefix}-{int(datetime.now(timezone.utc).timestamp())}"

    def _format_report_line(self, label: str, result: UpsertResult) -> str:
        """格式化单条操作报告行。"""
        return (
            f"- {label}: 新增 {result.added} 条，"
            f"更新 {result.updated} 条，删除 {result.deleted} 条"
        )

    async def terminate(self):
        """插件销毁时无需特殊处理。"""
