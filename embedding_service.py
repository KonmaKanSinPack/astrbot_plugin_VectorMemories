"""
Vector embedding service — 语义相似度记忆检索。
支持两种后端：AstrBot 内置 Embedding 服务商 / 手动配置 OpenAI 兼容 API。
纯 Python 余弦相似度，零额外依赖。
"""

import math
from typing import Any, Dict, List, Optional, Tuple

from astrbot.api import logger
from openai import AsyncOpenAI


class EmbeddingService:
    """封装 embedding 后端，提供统一的向量化与相似度排序接口。

    astrbot 模式每次调用时重新从 context 取 provider，避免缓存
    因 /reload-plugins 或连接池回收而失效的实例。
    manual 模式直接创建 AsyncOpenAI 客户端。
    """

    def __init__(
        self,
        provider_source: str = "manual",
        context: Any = None,
        provider_id: str = "",
        api_base_url: str = "https://api.openai.com/v1",
        api_key: str = "",
        model_name: str = "text-embedding-ada-002",
        dimensions: int = 1536,
    ):
        self.provider_source = provider_source
        self.model_name = model_name
        self.dimensions = dimensions
        self._context = context
        self._provider_id = provider_id
        self._client: Optional[AsyncOpenAI] = None

        if provider_source == "astrbot" and context is not None:
            self._ready = True
        elif provider_source == "manual" and api_key:
            self._client = AsyncOpenAI(api_key=api_key, base_url=api_base_url)
            self._ready = True
        else:
            self._ready = False

    @property
    def is_ready(self) -> bool:
        """后端可用时可发起调用。"""
        return self._ready

    # ------------------------------------------------------------------
    # 惰性获取 AstrBot provider（每次调用实时取，避免缓存过期实例）
    # ------------------------------------------------------------------

    def _get_astrbot_provider(self) -> Any:
        """每次调用时重新从 context 获取 provider。

        有指定 ID 时走 get_provider_by_id；否则取第一个 embedding provider。
        """
        # 有指定 ID → 精确查找
        if self._provider_id:
            get_by_id = getattr(self._context, "get_provider_by_id", None)
            if get_by_id:
                p = get_by_id(self._provider_id)
                if p is not None and hasattr(p, "get_embedding"):
                    return p

        # 无 ID 或精确查找失败 → 取第一个 embedding provider
        get_all = getattr(self._context, "get_all_embedding_providers", None)
        if get_all is None:
            return None
        providers = get_all()
        return providers[0] if providers else None

    # ------------------------------------------------------------------
    # Embedding helpers
    # ------------------------------------------------------------------

    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """返回单条文本的 embedding 向量，失败返回 None。"""
        if not self._ready or not text or not text.strip():
            return None
        if self.provider_source == "astrbot":
            return await self._embed_via_astrbot(text.strip())
        return await self._embed_via_openai(text.strip())

    async def get_embeddings(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        """批量文本向量化，返回与 texts 等长的列表。"""
        if not self._ready or not texts:
            return [None] * len(texts)
        if self.provider_source == "astrbot":
            return await self._embeddings_via_astrbot(texts)
        return await self._embeddings_via_openai(texts)

    # ------------------------------------------------------------------
    # Backend: AstrBot 内置
    # ------------------------------------------------------------------

    async def _embed_via_astrbot(self, text: str) -> Optional[List[float]]:
        provider = self._get_astrbot_provider()
        if provider is None:
            logger.warning("[VectorMemories] 惰性获取 embedding provider 为空")
            return None
        try:
            result = await provider.get_embedding(text)
            return list(result) if result is not None else None
        except Exception:
            logger.warning("AstrBot embedding provider failed", exc_info=True)
            return None

    async def _embeddings_via_astrbot(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        provider = self._get_astrbot_provider()
        if provider is None:
            logger.warning("[VectorMemories] 惰性获取 embedding provider 为空")
            return [None] * len(texts)

        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            return [None] * len(texts)
        indices, clean_texts = zip(*indexed)

        try:
            raw = await provider.get_embeddings(list(clean_texts))
        except Exception:
            logger.warning("AstrBot batch embedding failed", exc_info=True)
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        if isinstance(raw, list) and len(raw) == len(indices):
            for idx, emb in zip(indices, raw):
                results[idx] = list(emb) if emb is not None else None
        return results

    # ------------------------------------------------------------------
    # Backend: 手动 OpenAI 兼容 API
    # ------------------------------------------------------------------

    async def _embed_via_openai(self, text: str) -> Optional[List[float]]:
        try:
            resp = await self._client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
            )
            return list(resp.data[0].embedding)
        except Exception:
            logger.warning(
                "OpenAI embedding API call failed for single text", exc_info=True
            )
            return None

    async def _embeddings_via_openai(
        self, texts: List[str]
    ) -> List[Optional[List[float]]]:
        indexed = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
        if not indexed:
            return [None] * len(texts)
        indices, clean_texts = zip(*indexed)

        try:
            resp = await self._client.embeddings.create(
                model=self.model_name,
                input=list(clean_texts),
                dimensions=self.dimensions,
            )
        except Exception:
            logger.warning(
                "OpenAI batch embedding API call failed", exc_info=True
            )
            return [None] * len(texts)

        results: List[Optional[List[float]]] = [None] * len(texts)
        for data in resp.data:
            results[indices[data.index]] = list(data.embedding)
        return results

    # ------------------------------------------------------------------
    # Similarity & ranking（纯 Python，无 numpy）
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(a: List[float], b: List[float]) -> float:
        """两个等长向量的余弦相似度。

        zip 以较短者为准 → 维度不匹配时不会崩溃。
        空向量或零范数返回 0.0。
        """
        if not a or not b:
            return 0.0
        dot = norm_a = norm_b = 0.0
        for x, y in zip(a, b):
            dot += x * y
            norm_a += x * x
            norm_b += y * y
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return dot / (math.sqrt(norm_a) * math.sqrt(norm_b))

    def rank_memories(
        self,
        query_embedding: List[float],
        memories: List[Dict],
        top_k: int = 5,
        embedding_key: str = "embedding",
    ) -> List[Tuple[Dict, float]]:
        """按与 query_embedding 的余弦相似度对 memories 排序。

        无 embedding 的记忆记 0.0 分排在末尾。
        返回最多 top_k 个 (记忆字典, 相似度) 元组，降序排列。
        """
        if not memories:
            return []

        scored: List[Tuple[Dict, float]] = []
        for mem in memories:
            emb = mem.get(embedding_key)
            score = self.cosine_similarity(query_embedding, emb) if emb else 0.0
            # 相似度为 0 的记忆无意义（无 embedding 或纯正交），直接跳过
            if score > 0.0:
                scored.append((mem, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
