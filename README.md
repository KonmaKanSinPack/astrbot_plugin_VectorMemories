# 向量记忆系统 (VectorMemories)

可配置 embedding 模型的向量记忆插件。与 AstrBot 内置 Embedding 服务商无缝集成，基于语义相似度检索记忆，每次只返回最相关的 N 条。

## 与原版的区别

| | 原版 simple_memory | VectorMemories |
|---|---|---|
| 检索方式 | subject_id 精确匹配，返回全部记忆 | 语义相似度排序，只返回 top-K |
| 记忆量 | 随着积累不断膨胀，挤占 context | 始终控制在 top-K 条，节省 token |
| embedding | 无 | 支持 AstrBot 内置服务商 / 手动 API |
| 向下兼容 | — | 关闭向量模式即恢复原版行为 |

## 设计思路

### 数据流

```
用户消息 → embedding 模型 → 查询向量
                                    │
记忆库 ──→ subject_id 预过滤 ──→ 余弦相似度排序 ──→ top-K ──→ 注入 system_prompt
```

两步检索保证安全：
1. **subject_id 预过滤**：只取属于当前用户/群的记忆（隐私硬边界）
2. **余弦相似度排序**：越相关的越靠前，取前 K 条

### 向量存储

embedding 向量与记忆条目一起存在 JSON 文件中，每条记忆多一个 `embedding` 字段。不需要额外的向量数据库 —— 个人记忆量级下（< 500 条），纯 Python 遍历 + 余弦相似度计算 < 1ms。

### 向量生成时机

LLM 返回记忆更新 JSON → 解析并应用到内存 → 为新/变更的条目批量调用 embedding API → 二次持久化。只在记忆内容真正变化时才生成向量，避免重复调用。

### 失败降级

embedding API 调用失败、密钥未配置、网络超时 → 自动回退到 subject_id 精确匹配模式。核心功能不依赖向量，永远可用。

---

## 配置说明

在 AstrBot WebUI 的插件配置页面中设置以下选项：

### 基本开关

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `enable_vector_memory` | bool | false | 启用向量检索 |
| `retrieval_top_k` | int | 5 | 每种记忆类型返回的最相关条数 |
| `include_all_core_memory` | bool | true | core_memory 是否全部包含（建议开启） |

### Embedding 来源

| 配置项 | 类型 | 默认 | 说明 |
|--------|------|------|------|
| `embedding_provider_source` | 下拉 | `astrbot` | **astrbot**：使用管理面板中已配置的 Embedding 服务商；**manual**：手动填写下方 API |

#### 方式 A：AstrBot 内置服务商（推荐）

1. 在 AstrBot WebUI → 服务提供商 → 新增 → 选择 **Embedding**
2. 填写 API 信息（支持 OpenAI 兼容、Ollama 本地、Gemini 等）
3. 本插件配置中 `embedding_provider_source` 保持默认 `astrbot` 即可
4. 开启 `enable_vector_memory` → 完成

**推荐免费方案**：

| 提供商 | 模型 | 说明 |
|--------|------|------|
| PPIO 派欧云 | `baai/bge-m3` | 免费，国内直连 |
| Ollama 本地 | `nomic-embed-text` / `bge-m3` | 完全离线，无需 API Key |
| Google AI Studio | `gemini-embedding-2-preview` | 免费额度 |

#### 方式 B：手动配置 API

1. `embedding_provider_source` 选 `手动配置 API`
2. 填写 `embedding_api_base_url`、`embedding_api_key`、`embedding_model_name`、`embedding_dimensions`
3. 兼容所有 OpenAI API 格式的服务

---

## 使用方法

### /mem 命令

| 命令 | 说明 |
|------|------|
| `/mem gen` | 让 LLM 分析当前对话，生成记忆更新 JSON 并自动应用。加 `--full` 使用全部对话历史 |
| `/mem gen <额外指令>` | 在提示词末尾追加额外指令，如 `/mem gen 删除所有关于天气的记忆` |
| `/mem check` | 查看上次 `/mem gen` 返回的原始内容 |
| `/mem rebuild` | 备份当前记忆 → 清空 → LLM 基于旧记忆从零重构 |
| `/mem help` | 显示使用说明 |

**建议流程**：聊几轮 → `/mem gen` → LLM 总结并写入记忆 → `/mem check` 确认

### LLM 工具（LLM 自动调用）

插件向 LLM 暴露以下工具，LLM 在需要时会自动调用：

| 工具 | 用途 |
|------|------|
| `update_one_memory` | 增/改/删单条记忆 |
| `delete_several_memories` | 批量删除同类型记忆 |
| `search_memory_by_user_name` | 按用户名搜索记忆，支持语义查询 |
| `update_user_roster_id_dict` | 更新用户名 → subject_id 映射 |
| `check_user_roster_id_dict` | 查看当前所有映射关系 |

---

## 记忆类型说明

| 类型 | 用途 | 示例 |
|------|------|------|
| `core_memory` | AI 人格/价值观/表达风格（抽象、稳定） | "我倾向于用简洁的回答风格" |
| `long_term` | 长期知识、用户档案、可复用事实 | "用户小明是程序员，偏好 Python" |
| `medium_term` | 近期主题、阶段性任务、上下文 | "本周在讨论 AstrBot 插件开发" |

---

## 注意事项

- **不要随意更换 embedding 模型**：更换后向量维度可能不同，旧记忆的向量将无法与新查询匹配。如果确实要换，建议先 `/mem rebuild` 重建记忆。
- **subject_id 规则**：LLM 写入记忆时会把用户私有信息关联到具体用户 ID，不要手动改为 `global`。
- **core_memory 规则**：只存 AI 人格相关内容，不要往里塞具体事实。
