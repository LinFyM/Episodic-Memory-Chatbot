# SFT数据抽取流程详解

本文档详细说明第一步和第二步训练中所有涉及SFT数据抽取的地方及其执行流程。

## 第一步训练：`<recall>` token embedding训练

### 1. SFT向量提取阶段（训练前）

**位置**：`src/training/memory_extraction.py::extract_sft_vectors_for_recall_training()`

**执行时机**：在第一步训练开始前，从原始SFT数据集中提取思考段（`<think>...</think>`）的向量。

**流程**：
1. **计算目标数量**：`required_sft_count = num_memory_entries * 1.5`（记忆条目数的1.5倍）
2. **加载原始数据集**：调用 `_load_sft_dataset()` 加载所有SFT样本（约110000条）
3. **随机打乱**：`random.shuffle(sft_samples)` 打乱顺序
4. **逐条处理**：
   - 标准化messages：`_standardize_sft_messages(sample)`
   - 提取思考段：从完整文本中提取 `<think>...</think>` 之间的内容
   - **Token长度校验**：
     - 如果配置了 `sft_max_tokens`，使用tokenizer计算思考段的token数
     - 如果超过限制，**跳过该样本，继续处理下一个**
   - 收集合格的思考段文本到 `sft_thinking_texts`
   - 当收集数量达到 `required_sft_count` 时，**停止遍历**
5. **数量校验**：如果最终收集的思考段数量 < `required_sft_count`，抛出 `ValueError`
6. **批量提取向量**：调用 `_batch_extract_embeddings()` 将所有思考段文本转换为向量
7. **保存到临时文件**：保存为 `temp_sft_vectors.pt`，包含 `texts` 和 `embeddings`

**关键点**：
- ✅ **递归补充机制**：如果某条样本超长被跳过，会继续从剩余样本中抽取，直到凑够目标数量
- ✅ **一次性处理**：只在训练前执行一次，后续训练直接使用已提取的向量文件
- ✅ **Token限制**：在提取阶段就过滤掉超长的思考段，确保后续训练不会遇到超长数据

### 2. 训练阶段使用

**位置**：`src/training/training_service.py::train_recall_token()`

**执行时机**：第一步训练时，从 `temp_sft_vectors.pt` 加载已提取的SFT向量。

**流程**：
1. 加载 `temp_sft_vectors.pt`，获取 `sft_texts` 和 `sft_embeddings`
2. 随机抽取 `required_sft_count` 条向量（如果向量文件中的数量不足，会在提取阶段抛错，所以这里不需要min）
3. 与记忆条目向量合并，形成训练数据
4. **不再进行token长度校验**（已在提取阶段完成）

---

## 第二步训练：记忆解码训练

### 1. SFT采样器初始化（训练前）

**位置**：`src/training/training_service.py::train_memory_decoding()`

**执行时机**：在第二步训练开始前，准备SFT采样器供每个epoch使用。

**流程**：
1. **加载原始数据集**：调用 `_load_sft_dataset()` 加载所有SFT样本
2. **标准化处理**：
   - 遍历所有样本，调用 `_standardize_sft_messages(sample)` 标准化
   - 保存为 `standardized_sft_samples`，每个元素包含：
     - `messages`：标准化后的消息列表
     - `index`：在原始数据集中的索引
3. **创建采样器函数**：`_epoch_sampler(total_target)`
   - 闭包捕获 `standardized_sft_samples`、`preloaded_processor`、`sft_max_tokens`
   - 返回一个可调用对象，供每个epoch调用
   - **注意**：只保存标准化后的messages和索引，不占用显存（文本数据在内存中）

**关键点**：
- ✅ **不预抽取**：只准备采样器，不提前抽取数据
- ✅ **保留原始索引**：每个标准化样本都记录其在原始数据集中的位置，用于日志追踪
- ✅ **显存占用优化**：只保存标准化后的messages（文本数据）和索引（整数），不占用显存。110000条样本的messages和索引只占用内存，不会导致显存问题

### 2. 每个epoch的SFT抽样（训练中）

**位置**：`src/training/text_memory_train.py::MixedMemorySFTDataset.refresh_epoch_data()`

**执行时机**：每个epoch开始前，调用 `dataset.refresh_epoch_data()` 重新抽取SFT数据。

**流程**：
1. **调用采样器**：
   ```python
   sampled_full_texts, sampled_messages, all_sources = self.sft_epoch_sampler(
       total_target=int(memory_count * 1.5)  # 抽取1.5倍记忆条目数量的SFT样本
   )
   ```
2. **采样器内部执行**（`_sample_sft_for_epoch()`）：
   - 随机打乱候选索引：`random.shuffle(candidate_indices)`
   - 遍历打乱后的索引：
     - 获取标准化样本：`record = sample_records[idx]`
     - **Token长度校验**：
       - 调用 `_is_sft_within_token_limit()` 检查messages的token数
       - 如果超过 `sft_max_tokens`，**跳过该样本，继续下一个**
     - **收集样本**：
       - 将messages添加到 `selected_messages`
       - 提取完整文本（包含`<think>`段），如果存在则添加到 `selected_full_texts`，否则添加 `None`
       - 记录原始索引到 `selected_message_sources`
     - 当收集数量达到 `total_target` 时，**停止遍历**
   - **数量校验**：如果最终数量不足，抛出 `ValueError`
   - 返回：`(selected_full_texts, selected_messages, selected_message_sources)`
3. **三等分SFT样本**：
   - 将抽到的 `total_target` 条样本三等分：
     - **前缀SFT**：前1/3，用于记忆-前置训练
     - **夹心SFT**：中1/3，用于记忆-前后拼接训练（需要完整文本）
     - **纯SFT**：后1/3，用于纯SFT训练
4. **更新数据集**：
   - `self.sft_messages_list = prefix_messages + pure_messages`（前缀和纯SFT都用messages）
   - `self.memory_dataset.sft_full_texts = middle_full_texts`（夹心SFT的完整文本）
   - 保存原始索引：`self.current_sft_msg_source_indices`、`self.current_sft_full_source_indices`
5. **构造混合训练样本**：
   - 记忆-前置：记忆条目的1/3 + 前缀SFT（从 `prefix_messages` 中抽取）
   - 记忆-前后拼接：记忆条目的1/3 + 夹心SFT（从 `middle_full_texts` 中抽取）
   - 纯SFT：记忆条目的1/3 + 纯SFT（从 `pure_messages` 中抽取）
   - **随机打散**：所有混合样本构造完成后，调用 `random.shuffle(self.mixed_indices)` 打散顺序

**关键点**：
- ✅ **每个epoch重新抽样**：每次调用 `refresh_epoch_data()` 都会从原始数据集中重新随机抽取
- ✅ **递归补充机制**：如果某条样本超长被跳过，会继续从剩余样本中抽取，直到凑够目标数量
- ✅ **原始索引追踪**：日志中会显示每个样本在原始数据集中的真实索引，便于验证随机性

---

## 总结对比

| 阶段 | 执行时机 | 数据来源 | Token限制 | 递归补充 | 原始索引追踪 | 随机打散 |
|------|---------|---------|-----------|---------|-------------|---------|
| **第一步：向量提取** | 训练前一次 | 原始SFT数据集 | ✅ 提取时校验 | ✅ 是 | ❌ 否 | ❌ 否 |
| **第一步：训练使用** | 训练时 | 已提取的向量文件 | ❌ 无需（已过滤） | ❌ 否 | ❌ 否 | ✅ 是（合并后） |
| **第二步：采样器初始化** | 训练前一次 | 原始SFT数据集 | ❌ 仅标准化 | ❌ 否 | ✅ 是 | ❌ 否 |
| **第二步：每个epoch抽样** | 每个epoch开始 | 标准化样本池 | ✅ 抽样时校验 | ✅ 是 | ✅ 是 | ✅ 是（三等分后） |

## 关键设计原则

1. **避免一次性处理110000条数据**：
   - 第一步：只处理到凑够目标数量就停止
   - 第二步：每个epoch只抽取需要的数量，不预加载全部

2. **Token限制严格执行**：
   - 所有SFT数据在进入训练前都必须通过token长度校验
   - 超长样本会被跳过，通过递归补充机制确保数量充足

3. **随机性保证**：
   - 每次抽样前都会 `random.shuffle()` 打乱顺序
   - 每个epoch重新从原始数据集中抽样，确保不同epoch使用不同样本
   - 混合样本构造完成后，会再次 `random.shuffle()` 打散顺序，确保训练时不同类型样本随机出现
   - 日志中显示原始索引，便于验证随机性

4. **原始索引追踪**：
   - 第二步训练中，每个样本都记录其在原始数据集中的索引
   - 日志输出时会显示这些索引，方便确认每个epoch确实抽取了不同的样本

