# 向量记忆系统 (VectorMemories)

可配置 embedding 模型的向量记忆插件。与 AstrBot 内置 Embedding 服务商无缝集成，基于语义相似度检索记忆，每次只返回最相关的 N 条。

## 与旧版 simple_memory 的区别

| | 旧版 simple_memory | vector_memories |
|---|---|---|
| 检索方式 | subject_id 精确匹配，返回全部记忆 | long_term / medium_term 按语义相似度排序，只返回 top-K |
| core_memory | 返回全部 | **默认全部返回**（AI 人格记忆通常少且重要），可选也走 top-K |
| 记忆量 | 随着积累膨胀，挤占 context | lt / mt 始终控制在 top-K 条 |
| embedding 来源 | 无 | AstrBot 内置服务商 或 手动 API |

## 设计思路

### 检索数据流

core_memory 和 lt/mt 走两条不同的路径：

```
                    用户消息
                       │
                       ▼
              embedding 模型 → 查询向量
                       │
         ┌─────────────┴─────────────┐
         ▼                           ▼
   core_memory                long_term + medium_term
         │                           │
   include_all_core?          subject_id 预过滤
   ├─ true → 全部返回              │
   └─ false → top-K          余弦相似度排序
                                  │
                              取 top-K
                                  │
         └─────────────┬─────────────┘
                       ▼
              拼接文本 → 注入 system_prompt
```

两步检索（仅 lt / mt）：
1. **subject_id 预过滤** — 只取属于当前用户/群/global 的记忆（**隐私硬边界**，绝不跨用户）
2. **余弦相似度排序** — 越相关的越靠前，取前 K 条

### core_memory 为什么默认全部返回

core_memory 存的是 AI 人格（"我倾向于简洁回答"这类），数量通常 < 10 条，全部塞进 prompt 成本极低。而且 AI 人格应该始终生效，不应因话题变化而被过滤。如果想让它也走 top-K，设置 `include_all_core_memory = false`。

### 向量存储

embedding 向量与记忆条目一起存在 JSON 文件中，每条记忆多一个 `embedding` 字段。不需要额外的向量数据库 —— 个人记忆量级下（< 500 条），纯 Python 遍历 + 余弦相似度计算 < 1ms。

### 向量生成时机（写路径）

```
/mem gen → LLM 分析对话 → 返回 JSON（增/改/删）
    → _apply_operations() 应用到内存
    → save() 持久化（无 embedding）
    → _embed_new_entries() 批量调 embedding API
    → save() 二次持久化（带 embedding）
```

只在 `content` 真正变化时才重新生成向量（`_upsert_and_delete()` 中检测到内容变更会清除旧 embedding，触发 `_embed_new_entries()` 重新生成）。

### 失败降级

embedding API 调用失败 / 密钥未配置 / 网络超时 / AstrBot 内置 provider 找不到 → 自动回退到 subject_id 精确匹配模式。核心功能不依赖向量。

---

## 配置说明

在 AstrBot WebUI 的插件配置页面中设置。

### 基本配置

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `enable_vector_memory` | bool | false | 总开关：是否启用向量检索 |
| `retrieval_top_k` | int | 5 | 每种记忆类型（lt / mt）最多返回的条数 |
| `include_all_core_memory` | bool | true | core_memory 是否全部返回（建议开启） |
| `use_global` | bool | true | true = 所有用户共享一个记忆文件；false = 每人独立文件 |

### Embedding 来源

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `embedding_provider_source` | 下拉 | `astrbot` | `astrbot` = 使用管理面板已配好的服务商；`manual` = 手动填下方字段 |

### 方式 A：AstrBot 内置服务商（推荐，默认）

1. AstrBot WebUI → 服务提供商 → 新增 → 选 **Embedding** → 填 API 信息
2. 本插件配置中 `embedding_provider_source` 保持默认 `astrbot`
3. 开启 `enable_vector_memory` → 完成，无需重复填密钥

**推荐免费方案**：

| 提供商 | 模型 | 说明 |
|--------|------|------|
| PPIO 派欧云 | `baai/bge-m3` | 免费 1024 维，国内直连 |
| Ollama 本地 | `nomic-embed-text` / `bge-m3` | 完全离线 |
| Google AI Studio | `gemini-embedding-2-preview` | 免费额度 |

### 方式 B：手动配置 API

`embedding_provider_source` 选 `手动配置 API` 后，填写以下字段：

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `embedding_api_base_url` | string | `https://api.openai.com/v1` | API 地址 |
| `embedding_api_key` | string | 空 | API 密钥 |
| `embedding_model_name` | string | `text-embedding-ada-002` | 模型名 |
| `embedding_dimensions` | int | 1536 | 向量维度 |
| `mem_prompt` | text | 空 | 追加到记忆更新提示词末尾的额外指令 |

---

## 使用方法

### /mem 命令

| 命令 | 说明 |
|------|------|
| `/mem gen` | 让 LLM 分析对话 → 生成记忆更新 JSON → **自动应用并持久化**。加 `--full` 使用全部对话历史 |
| `/mem gen <额外指令>` | 在提示词末尾追加额外指令，如 `/mem gen 删除所有关于天气的记忆` |
| `/mem check` | 查看上次 `/mem gen` 返回的原始 JSON 内容 |
| `/mem rebuild` | 备份当前记忆文件 → 清空 → LLM 基于旧记忆从零重构 |
| `/mem test` | 验证 embedding provider 是否正常（获取 provider + 测试调用 get_embedding） |
| `/mem help` | 显示使用说明 |

**建议流程**：聊几轮 → `/mem gen`（LLM 自动总结 + 写入 + 生成向量）→ `/mem check` 确认

### LLM 工具（LLM 在对话中自动调用）

插件向 LLM 暴露以下工具：

| 工具名 | 用途 |
|--------|------|
| `update_one_memory` | 增 / 改 / 删单条记忆 |
| `delete_several_memories` | 批量删除同类型记忆 |
| `search_memory_by_user_name` | 按用户名搜索记忆（支持 `query` 参数做语义检索） |
| `update_user_roster_id_dict` | 更新 user_name → subject_id 映射 |
| `check_user_roster_id_dict` | 查看当前所有映射 |

---

## 记忆类型说明

| 类型 | 用途 | 数量特点 | 示例 |
|------|------|---------|------|
| `core_memory` | AI 人格 / 价值观 / 表达风格 | 少（< 10 条），稳定 | "我倾向于简洁直接的回答" |
| `long_term` | 用户档案 / 长期知识 / 可复用事实 | 持续增长 | "用户小明是后端，偏好 Python" |
| `medium_term` | 近期主题 / 阶段性任务 | 中等，可能过期 | "这几天在讨论 AstrBot 插件开发" |

---

## 注意事项

- **不要随意更换 embedding 模型**：模型变 → 维度可能变 → 旧向量的余弦相似度无意义。如果要换，先 `/mem rebuild` 重建记忆。
- **subject_id 隔离**：LLM 写入记忆时自动把用户私人信息关联到具体 subject_id，不会写成 `global`。
- **core_memory 约束**：只存 AI 人格层面的抽象内容，具体事实放 lt 或 mt。
- **向量模式关闭时**：行为完全等同于旧版（simple_memory），无任何影响。
