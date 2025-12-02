# 自回归记忆统一方案（供 LLM 深度阅读）

> **目标**：将记忆检索机制与标准自回归生成实现“完全对称”的融合，并配套一套能把记忆写入模型的训练体系。本文不讨论任何具体业务场景，只聚焦通用可复用的技术细节。

---

## 1. 引言：记忆是通往持续智能的必要能力

1. **Yann LeCun 的观点**  
   - 现代深度学习系统虽然庞大，但本质仍是“函数逼近器”，缺乏可以读写的记忆。  
   - 若模型无法把新经验写入可检索的存储区，就很难实现持续学习。  
   - 因此，一个能在自回归过程中“主动读写记忆”的机制，是迈向真正自主系统的关键。

2. **Ilya Sutskever 的观点**  
   - “超级智能应该像永远在学习的 15 岁少年，而不是能执行任意指令的工人。”  
   - 少年式智能意味着：随时吸收新知识、在对话中调取旧经验，并把结果以自然语言输出。  
   - 这种行为要求模型在自回归循环内部就能触发记忆检索，而不是依赖外部工具。

3. **现有方案的不足**  
   | 方案 | 优点 | 典型问题 |
   | ---- | ---- | -------- |
   | 额外 SFT 微调 | 结构简单 | 训练成本高；新知识写入慢；推理时无法动态选择记忆 |
   | RAG | 不改模型参数 | 推理链路复杂；取回的文本难与模型原有思路对齐；需要额外的 prompt 工程 |

   本方案希望以“极低的推理开销 + 接近 RAG 的灵活性”填补两者之间的空隙。

---

## 2. 总览：在自回归循环里“召唤记忆”

### 2.1 关键概念

| 名称 | 说明 |
| --- | --- |
| `<recall>` token | 模型在自回归过程中自主生成，表示“我要调取记忆” |
| Memory Head | 与 `lm_head` 对称的模块：输入 `<recall>` 的 hidden state，输出所有记忆向量的 logits |
| Memory Embedding | 与 `input_embeddings` 对称的模块：把选定的记忆向量直接作为下一步输入 embedding |
| `<|memory_pad|>` | 占位符 token，用来占据“记忆注入”的位置，训练和推理都保持一致 |

### 2.2 整体流程（可结合 `docs/MEMORY_RECALL_FLOW.md`）

1. **触发**：模型生成 `<recall>`。  
2. **摘要**：取 `<recall>` 位置的 hidden state 作为 query，代表当前上下文。  
3. **检索**：在记忆库中计算 query 与所有向量的余弦相似度，得到 logits。  
4. **采样**：与普通 token 完全一致，使用 `TemperatureLogitsWarper`、`TopKLogitsWarper`、`TopPLogitsWarper` 处理，再采样或贪婪选择一个记忆条目。  
5. **注入**：强制下一步生成 `<|memory_pad|>`，同时把选中的记忆向量当作 embedding 输入模型。  
6. **生成**：模型以记忆 embedding 为起点继续自回归，输出被训练好的“记忆文本 + 结束提示 + 后续 SFT 内容”。  
7. **恢复**：完成记忆输出后，所有标志位被清理，模型继续普通自回归。

### 2.3 对称设计速览

| 维度 | 普通 token 生成 | 记忆扩展 |
| ---- | --------------- | -------- |
| Head | `lm_head` 输出 vocab logits | **Memory Head** 输出记忆 logits |
| 采样 | logits_processor → logits_warper → softmax | 完全相同 |
| 输入 | token ID → embedding lookup | 直接注入记忆向量（跳过 embedding 层） |
| 控制位 | 无 | `<recall>` 触发 / `<|memory_pad|>` 占位 |
| 状态管理 | KV cache、停止条件等 | 复用，无额外分支 |

这种“对称性”是方案的核心：记忆流程看起来像是模型在生成“一个特殊的 token”，因此极易融入现有的自回归实现。

---

## 3. 深入：每个环节的技术细节

### 3.1 `<recall>` 触发判定
1. 每次循环开始前，`custom_generate` 会调用 `model.prepare_inputs_for_generation()`，得到当前的 `input_ids`。  
2. 若 `input_ids` 的最后一个 token 等于 `<recall>` 且 `memory.autoregressive_recall.enabled` 为真，则设置 `recall_triggered = True`，进入记忆分支。  
3. 如果不满足条件，流程与普通自回归完全相同。

### 3.2 Memory Head：从 `<recall>` hidden state 到记忆 logits

参考 `docs/AUTOREGRESSIVE_GENERATION.md`：
1. **Query 构造**  
   ```python
   query_vector = last_hidden_state[0, -1, :]  # 即 <recall> 位置的 hidden state
   ```
2. **相似度计算**  
   - 记忆库中的所有向量在写入时都做了 L2 normalize。  
   - Memory Head 简单地做一次矩阵乘（query vs. 全量记忆向量），得到所有候选的余弦分数。  
   - 不做 softmax，只返回 logits（形状 `[num_memory_entries]`），与 `lm_head` 输出 vocab logits 完全对称。  
3. **调试输出**：可选地打印相似度 top-k，便于调试记忆内容。

### 3.3 Logits 统一处理
- 记忆 logits 与 token logits 使用同样的 `LogitsProcessorList`（可配置重复惩罚等）与 `LogitsWarper`（Temperature / TopK / TopP）。  
- 之所以重复 Top-k 截断，是为了确保记忆采样与 token 采样的行为完全一致。  
- `autorecall_use_sampling` 控制是否采样；若为 False，则直接 argmax。

### 3.4 Memory Embedding：跳过 embedding 层
1. 把选中的记忆向量 reshape 成 `[1, 1, hidden_dim]`，并复制到模型所在的 device / dtype。  
2. 设置 `forced_next_token_id = memory_pad_token_id`，确保自回归序列仍然有一个 token ID。  
3. 下一轮前向时，如果 `override_next_embed` 非空，就从 `forward_inputs` 中删除 `input_ids`，改传 `inputs_embeds=override_next_embed`。  
4. 这意味着 `<|memory_pad|>` 的 embedding 完全由记忆向量决定，而不是从词表 lookup。  
5. 前向完成后立即清除 `override_next_embed` 与 `forced_next_token_id`，避免污染后续循环。

### 3.5 循环时序（详见 `docs/MEMORY_RECALL_FLOW.md`）

| 循环 | 输入末尾 | 操作 | 状态变化 |
| ---- | -------- | ---- | -------- |
| **N** | 普通 token → 生成 `<recall>` | 无 | 常规自回归 |
| **N+1** | `<recall>` | Memory Head 检索 + 设置注入参数 | `override_next_embed`、`forced_next_token_id` 被填充 |
| **N+2** | `<|memory_pad|>` | 用记忆 embedding 前向；清空标志 | 所有记忆相关变量恢复默认 |

---

## 4. 训练体系：把记忆写进模型

### 4.1 核心目标
1. **Step 1：Recall Token 训练**  
   - 目的是让 `<recall>` 的 hidden state 成为“上下文摘要器”。  
   - 训练后 `<recall>` 应该能在不同上下文中产生稳定的 query 以驱动检索。

2. **Step 2：Memory Decoding 训练**  
   - 目的是让模型看到记忆向量就能还原原文，并保持原有的思维/对话结构。  
   - 同时需要保证模型不会因为频繁回忆而丢失普通对话能力。

### 4.2 Step 1 详解（参见 `docs/SFT_SAMPLING_FLOW.md`）

1. **样本来源**  
   - 从大规模 SFT 语料中提取 `<think>...</think>` 思考段。  
   - 目标数量 = 记忆条目数量 × 1.5（冗余用于过滤）。  
   - 抽样时打乱顺序，确保覆盖面。

2. **长度过滤**  
   - 使用 tokenizer 对每个 `<think>` 段落编码，若 token 数超过 `training_config.sft_max_tokens`（例如 2500），直接跳过。  
   - 跳过后立即从剩余样本中补抽，保证最终数量满足要求。

3. **训练方式**  
   - 输入：原始上下文（包含 `<think>` 段）。  
   - 目标：让模型在 `<recall>` 位置输出能够代表该段思考内容的 embedding。  
   - LoRA 仅挂在 Q/V 上，减少显存占用。  
   - 训练完成后得到 `<recall>` 的“聚合能力”，为 Step 2 打基础。

### 4.3 Step 2 详解（参见 `docs/SECOND_STEP_TRAINING_DATA.md`）

1. **数据集结构：MixedMemorySFTDataset**

| 类型 | 输入序列 | label 分布 | 设计动机 |
| ---- | -------- | ----------- | -------- |
| **memory_front** | 随机上下文 + `<recall>` + `<|memory_pad|>` + 记忆文本 + 结束提示 | 记忆文本 + 结束提示 | 训练模型在噪声环境下回忆 |
| **memory_full** | SFT 前缀 + `<recall>` + `<|memory_pad|>` + 记忆文本 + 结束提示 + SFT 后缀 | 记忆文本 + 结束提示 + SFT 后缀 | 让回忆内容与思维/回答结构无缝衔接 |
| **sft_only** | 纯 SFT 对话（无记忆 token） | 仅 assistant 段落 | 保持常规对话能力，避免退化 |

2. **label 设置要点**
   - `<recall>` token 必须有 label（等于其 token ID），确保触发能力不会逐渐丧失。  
   - `<|memory_pad|>` label 设为 -100，因为该位置在推理时由记忆向量覆盖。  
   - `memory_full` 的后缀来自同一条 SFT 样本的 `<think>` 之后部分，让模型在回忆结束后自然接回原问题的回答。  
   - `sft_only` 只训练 assistant 内容，system/user 均置为 -100。

3. **混合比例与抽样**
   - 每个 epoch 都调用 `refresh_epoch_data()`，重新从原始 SFT 池中抽样，总量约为记忆条目 × 1.5。  
   - 抽取后按三等分划分到三种类型，保证每个 batch 中长期混合三种样本。  
   - 抽样前会调用 `_is_sft_within_token_limit()`，使用 `processor.apply_chat_template(..., tokenize=True)` 计算真实 token 数，超限则跳过并补抽。  
   - 日志会输出原始索引，方便验证是否真的“每个 epoch 重抽”。

4. **训练循环**
   - `EnhancedTextMemoryTrainer` 使用 Accelerate（BF16 + 梯度累积），在多卡上保持效率。  
   - 训练过程中打印样本类型、embedding 插入位置、上下文长度等，方便定位异常。  
   - 每个 epoch 结束调用 `test_memory_recall()`，直接用真实记忆向量进行生成验证（确保 `<|memory_pad|>` 位于 `len(full_input_tokens) - 1`）。  
   - 训练完成后合并 LoRA 权重，得到最终部署模型。

### 4.4 为什么要这样设计？

1. **Step 1**：若不先训练 `<recall>`，模型会把 `<recall>` 当作普通 token，输出的 hidden state 质量极低，导致检索完全靠运气。  
2. **Step 2**：若只训练 memory_front，则模型在回忆后容易“结束输出”；memory_full 提供了“回忆 + 原 SFT 后缀”的组合，强迫模型回忆后继续按照原先思维/回答结构输出。  
3. `sft_only` 的存在防止模型只剩“回忆”这条出路，保证普通对话能力不退化。

---

## 5. 数据采样与长度控制：细致到每一步

### 5.1 Step 1：SFT `<think>` 抽样

| 步骤 | 描述 |
| ---- | ---- |
| 加载语料 | `_load_sft_dataset()` 返回原始 JSON 行 |
| 标准化 | `_standardize_sft_messages()` 把多模态内容收敛为统一格式 |
| 随机打乱 | `random.shuffle(sft_samples)` |
| 提取 `<think>` | `processor.apply_chat_template(..., tokenize=False)` → 查找 `<think>...</think>` |
| 长度校验 | `tokenizer(thinking_content, return_tensors="pt")`，超限跳过 |
| 收集 | 直到 `num_memory_entries * 1.5` 条合格样本 |
| 批量向量化 | `_batch_extract_embeddings()`，取最后一个 token 的 hidden state |

### 5.2 Step 2：按 epoch 动态抽样

1. 仅在训练开始前把 SFT 样本标准化为 `{messages, index}`，不做抽样。  
2. 每个 epoch 调用 `_sample_sft_for_epoch(total_target)`：  
   - `candidate_indices = list(range(len(sample_records)))` → `random.shuffle()`  
   - 逐条检查 token 长度：  
     ```python
     within_limit, seq_len = self._is_sft_within_token_limit(processor, messages, max_tokens)
     ```  
   - 合格样本记录 `messages`、`full_text`（若包含 `<think>`）、原始 `index`。  
   - 达到 `total_target` 后停止，否则抛错提醒“样本不足”。  
3. `refresh_epoch_data()` 把采样到的 SFT 三等分为：  
   - `prefix_messages`（记忆前置）  
   - `middle_full_texts`（记忆夹心，需要完整文本）  
   - `pure_messages`（纯 SFT）  
4. 记忆条目同样被随机打散并分配给 memory_front / memory_full / sft_only。  
5. 日志中打印“纯 SFT 原始索引”“夹心样本原始索引”，方便人工核查随机性。

这一系列步骤保证了“每个 epoch 都从原始语料重新抽取 + 严格长度控制 + 可追溯原始索引”。

---

## 6. 预期优势与技术定位

### 6.1 相较于常规微调
- **推理成本几乎为零**：回忆只是在输入里多了 `<recall>` + 一个向量，不需要更长 prompt 或额外模型。  
- **训练成本可控**：LoRA + 两阶段训练即可，把记忆写入模型而无需重新做大规模 SFT。  
- **灵活写入**：记忆条目来自任何来源，写入后即可在推理中触发，不受 SFT 语料限制。

### 6.2 相较于 RAG
- **推理链路更短**：不需要外部检索服务，也不需要额外 prompt 拼接。  
- **输出更连贯**：记忆内容已经通过第二阶段训练写入模型，因此能与原有思维流/回答风格自然衔接。  
- **知识真正“内化”**：虽然训练目标是“根据记忆 embedding 还原文本”，但模型在学习过程中同样会吸收文本中 token 之间的关系。

---

## 7. 下一步实验计划

| 阶段 | 内容 |
| ---- | ---- |
| **任务选择** | 选取一个对领域记忆敏感的任务（如专业知识问答、企业规章答疑）。 |
| **方案对比** | 本方案 vs 标准微调（SFT） vs RAG。 |
| **评估指标** | 记忆保持（多轮对话稳定性）、推理成本（延迟/显存）、输出结构（是否保持原思考链）。 |
| **期望** | 本方案在记忆保持上接近 RAG，但推理成本接近常规微调；同时输出结构最好。 |

---

## 8. 阅读指引与交叉引用

- 若需源码级细节：  
  - `docs/AUTOREGRESSIVE_GENERATION.md`: Memory Head、override_next_embed、KV cache 行为  
  - `docs/MEMORY_RECALL_FLOW.md`: N → N+1 → N+2 的时序图与状态机
- 若需数据构造与采样细节：  
  - `docs/SFT_SAMPLING_FLOW.md`: 两个阶段的 SFT 抽样策略  
  - `docs/SECOND_STEP_TRAINING_DATA.md`: memory_front / memory_full / sft_only 的具体 label 设计

> **给其他模型的提示**：  
> “请根据本文描述，梳理 `<recall>` 触发 → 记忆检索 → 记忆注入 的具体步骤，并说明两阶段训练分别解决了哪些问题。”

---

## 9. 总结

本方案在不改变模型主干结构的情况下，把“记忆检索”当作“生成一个特殊 token”来看待，从而：
1. 保证自回归循环的完整性和对称性；
2. 保证训练与推理的数据/行为完全一致（含 `<recall>`、`<|memory_pad|>` 的位置、embedding 注入方式等）；
3. 通过两阶段训练，让 `<recall>` 学会输出高质量 query，记忆向量也能驱动模型生成结构化回答。

下一步将通过系统实验对比普通微调与 RAG，验证该方案在成本、记忆保持、输出结构之间的平衡点。***

