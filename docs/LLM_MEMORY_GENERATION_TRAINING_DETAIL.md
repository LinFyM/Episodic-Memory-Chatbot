# 萝卜子 QQ 机器人自回归记忆生成 & 训练详案（供 LLM 阅读）

> **范围**：仅覆盖在线自回归生成流程（含记忆触发、KV cache 行为）与完整训练流水线（记忆提取 → 向量化 → Mixed Memory + SFT → 两步 LoRA）。排除 API / QQ 客户端层。

---

## 1. 在线生成：从历史到输出

### 1.1 Chat History 结构
- 每条消息记录：
  ```json
  {
    "role": "user" | "assistant",
    "content": [
      {"type": "text", "text": "..."},
      {"type": "image", "image": "http://..."} // 可选
    ],
    "timestamp": 17325xxxxx.123
  }
  ```
- 维护逻辑（`chat/history_manager.py`）：
  - `maintain_chat_history()` 去重（基于 role + timestamp + text），并按 `config.chat_history.max_history_length` 截断。
  - 超出部分异步写入 `memory.training.chat_history_storage_dir`，供训练阶段使用。

### 1.2 生成入口（`chat/reply_handler.generate_reply`）
1. **System Prompt 构造**：`chat/prompting.py::build_system_prompt` 汇总角色设定、记忆说明、输出格式等；由 `configs/prompts.yaml` 提供文本模板。
2. **历史拼接**：`full_messages = [system_prompt] + history`。
3. **Token 截断**：`truncate_history_by_tokens(processor, chat_history, system_prompt, max_tokens)`：
   - 调用 `processor.apply_chat_template(..., tokenize=True, add_generation_prompt=True)` 计算当前 token 数。
   - 若超出 `config.chat_history.max_input_tokens`（默认 32000），按 FIFO 移除历史消息并异步保存被移除段。
4. **序列化**：再调用一次 `processor.apply_chat_template(..., return_tensors="pt")` 得到 `input_ids`/`attention_mask`，发送给 `custom_generate`。

### 1.3 `custom_generate` 核心流程（`src/chat/generate.py`）
1. **准备阶段**：
   - 推导 `logits_processor`（重复惩罚等）、`logits_warper`（来自 `generation` 配置）。
   - 通过 `_update_model_kwargs_for_generation()` 维护 `past_key_values`。
2. **循环**（每个 token）：
   1. **回忆触发检测**：检查 `current_input_ids` 的最后一个 token 是否等于 `<recall>`（`recall_token_id` 由模型 tokenizer 转换），并确保 `memory.autoregressive_recall.enabled=true`。
   2. **前向传播**：
      ```python
      forward_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)
      outputs = forward_backbone(model, **forward_inputs, use_cache=True, output_hidden_states=True)
      last_hidden_state = ensure_last_hidden_state(outputs)
      ```
   3. **Memory Head**（若触发）：
      - query 向量：`query_vector = last_hidden_state[0, -1, :]`（即 `<recall>` 位置的 hidden state）。
      - `memory_head(query_vector, memory_db)`：
        - 强制把 `query_vector`、记忆库向量都转 bf16 & L2 normalize（在 `memory/vector_db.py` 中 add 时已 normalize）。
        - 计算余弦相似度 `torch.matmul(query_normalized, embeddings.T)`，**对所有记忆向量**求分数。
        - 不做 softmax，只返回 logits（`memory_logits`）与候选列表（包含 embedding / score / index）。
   4. **记忆 logits 处理**：
      ```python
      memory_warper = LogitsProcessorList([
          TemperatureLogitsWarper(autorecall_temperature),
          TopKLogitsWarper(autorecall_top_k),
          TopPLogitsWarper(autorecall_top_p)
      ])
      memory_scores = memory_warper(dummy_ids, memory_logits.unsqueeze(0)).squeeze(0)
      ```
      - `dummy_ids` 只是placeholder，因为 HF 的 `LogitsProcessor` 需要 input_ids。
      - 采样或贪婪取 `choice_idx` 取决于 `autorecall_use_sampling`。
   5. **记忆向量注入**：
      - 设置 `forced_next_token_id = memory_pad_token_id`，并记录 `memory_injection_positions.append((len(input_ids), memory_score))`。
      - `override_next_embed = memory_embedding(selected["embedding"], model)`：
        - reshape 到 `[1,1,hidden_dim]`，同步到模型 dtype/device。
      - 在下一轮前向时，若 `override_next_embed` 非空，则从 `forward_inputs` 中删 `input_ids`，改传 `inputs_embeds=override_next_embed`（避免 HF 报错“input_ids 与 inputs_embeds 只能选一个”）。
   6. **普通 token 生成**：
      - 获取 `outputs.logits[:, -1, :]`，套 `logits_processor` + `logits_warper`。
      - 若 `forced_next_token_id` 不为空，则绕过采样/贪婪直接输出 `<|memory_pad|>`。
      - 将新 token append 到 `input_ids`；`model_kwargs` 中的 `past_key_values` 会在 `_update_model_kwargs_helper` 中更新。
3. **KV Cache 行为**：
   - 在 `forward_backbone(..., use_cache=True)` 时，HF 初始化 `past_key_values`，尺寸与“输入 token 数 + `max_new_tokens`”有关；由于 `configs.generation.max_new_tokens=40960`，第一次推理时会预留相当大的显存。
   - 该 cache 由 CUDA allocator 持有，后续即使没有请求也保持占用，以便下一次生成复用。

---

## 2. 训练流水线

### 2.1 调度（`memory/training_scheduler.py`）
1. `train_job()`：
   - 获取 scheduler 锁 → 设置 `server_state.is_training=True`（API 层会拒绝新消息）。
   - 调 `MemoryTrainingService.run_training()`，捕获异常 → 释放锁。
2. 支持 `auto_restart_after_training`：
   - 若 true，则训练完成后调用 `training_service.cleanup_after_training()`，再根据 `restart_mode`（默认 `restart_server`）启动新的服务进程。

### 2.2 MemoryTrainingService 阶段划分

#### 阶段 A：准备
1. `_detect_project_root()`、`_prepare_output_dir()` 等确保目录存在。
2. `_ensure_training_modules_loaded()` 导入 `text_embedding_train` / `text_memory_train` / `memory_extraction` 等；（此前 `Tuple` 未导入导致 `NameError`，已修复）。
3. 清理显存：若设置了 `CUDA_VISIBLE_DEVICES`，只清理 `cuda:0`；否则遍历 `torch.cuda.device_count()`。

#### 阶段 B：记忆提取（`memory_extraction.py`）
1. **加载聊天历史**：
   - 从 `memory.training.chat_history_storage_dir` 扫描 `*.json`，各自包含 `{ "messages": [ ... ] }`。
   - `_standardize_sft_messages` 保留 `role + text`；多模态内容会被保留到 `content` 列表。
2. **构造多层 prompt**：
   - System prompt = `prompts.memory_extraction.system_prompt` + role_playing（可选）+ child depth 提示（递归时）。
   - User prompt：`memory_extraction.user_instruction`（默认“请开始提取记忆条目”）。
3. **生成记忆文本**：
   - `processor.apply_chat_template(full_messages, add_generation_prompt=True)` → `model.generate()`（配置 `max_new_tokens` 等）。
   - `_strip_formal_reply()` 取 `<think>` 之后内容；`_parse_memory_entries()` 过滤出记忆句子。
4. **写入临时文件**：`_append_memory_text_to_file()` 将文本保存到 `temp_memory_texts.pt`，便于分批处理。
5. **批量向量化**：
   - `_batch_extract_embeddings()` 按 `embedding_batch_size`（默认 8）将记忆文本嵌入，使用 `memory_vectorization.summary_prompt_template`。
   - tokenizer → 模型 forward → `last_token_idx = attention_mask.sum() - 1` → 取 `[layer=last, token=last]` hidden state → detach.cpu()。
   - 记录 batch 日志，并定期 `torch.cuda.empty_cache()`。
   - 输出 `texts`, `embeddings`，合并保存到 `temp_training_data.pt`。

#### 阶段 C：两步训练

1. **Recall Token 训练（`text_embedding_train.py`）**
   - 目标：让 `<recall>` + `<|memory_pad|>` 触发时更稳定，训练数据来自 `temp_training_data.pt`（embedding + 原文本）。
   - 调用 LoRA 适配，`LoraConfig(step1)` 只挂到 Q/V，减少显存。
   - 训练完成后保存 LoRA 权重，供下一阶段载入。

2. **Memory Decoding 训练（`text_memory_train.py`）**
   - **MixedMemorySFTDataset**：
     - `refresh_epoch_data()` 构造1:1:1索引：
       ```python
       half = memory_count // 2
       # 类型1: memory_front
       memory_type1_indices = sample(range(memory_count), half)
       # 类型2: memory_full (夹心)
       memory_type2_indices = sample(剩余)
       # 类型3: sft_pure = memory_count
       sft_pure_indices = sample(range(len(sft)), memory_count)
       ```
     - `_get_memory_decode_sample()`：拼 `context_tokens + activation_prompt + <recall> + <|memory_pad|>`，labels 对 `<recall>`、记忆文本、`</recall>`, `end_prompt` 打开；其余为 `-100`。
     - `_get_sft_sample_with_suffix()`：SFT 前后夹心场景；在 `tail_text` 中拼接后缀，确保 labels 覆盖后缀。
     - `_get_sft_sample()`：纯 SFT；`labels = input_ids.clone()`，随后 `labels[:assistant_start] = -100` 只训练助手输出。
   - **collate_fn**：
     - 若 `is_sft=True`，`embedding_position=-1`，`embeddings_to_insert` 用零向量占位。
     - 否则拼接 target label 与 embedding。
   - **训练循环（`EnhancedTextMemoryTrainer`）**：
     - Accelerator 负责混合精度（BF16）与梯度累积。
     - 训练开始时打印 sample_type、embedding_injection 位置。
     - 每 epoch 可根据 `training_config.memory_epochs` 迭代；结束后 `_run_test_generation()` 抽样验证，并带上 `guide_text`。
   - 结束后 `merge_and_save_model()` 生成最终 LoRA 合并模型（可选保存完整 VL assets）。

### 2.3 输出 / 清理
- 保存路径：
  - `memory.training.trained_model_dir = ./models/trained`（会生成 `model_YYYYMMDD_HHMMSS`）。
  - `memory.training.token_added_model_dir = ./models/token_added`（训练中若新增 token，会写拷贝）。
  - `memory.training.memory_db_dir = ./models/memory_db`（embedding 数据、`memory_embeddings.pt` 等）。
- `cleanup_after_training()`：
  - 删除聊天 JSON、临时训练数据、`uploads/` 下临时文件。
  - 保留 `memory_embeddings.pt` 等最终成果。

---

## 3. 关键参数摘录

| 组件 | 参数 | 默认 | 备注 |
| --- | --- | --- | --- |
| Generation | `max_new_tokens` | 40960 | 决定 KV cache 预留；若显存紧张可调至 1024/2048 |
| Generation | `temperature/top_p/top_k` | 1.0 / 0.95 / 20 | 普通 token 采样 |
| Recall | `autorecall_top_k` | 10 | 用于 memory logits warper |
| Recall | `autorecall_temperature/top_p` | 0.8 / 0.95 | 记忆采样温度/TopP |
| Mixed Dataset | 3 类比例 | 1:1:1 | memory_front / memory_full / sft_pure |
| Training | `embedding_batch_size` | 4 | 批量向量化时的 batch |
| Training | `memory_dataset_max_length` | 3000 | 构样时最大 token 长度 |
| Training | `memory_epochs` | 30 | 第二阶段训练 epoch，可按资源调小 |
| Training | `learning_rate` | 1e-4 | AdamW 学习率 |

---

## 4. LLM 消化提示
> 如需让其他模型理解此框架，可附一句指令：  
> “请根据文档中的步骤，梳理自回归记忆生成与训练流程的关键输入/输出、配置依赖和潜在瓶颈，并列出注意事项。”

--- 

附：如对内存、回忆行为有额外问题，可结合 `docs/AUTOREGRESSIVE_GENERATION.md` 和 `docs/MEMORY_RECALL_FLOW.md` 获取图示化解释。*** End Patch***
*** End Patch***}}} />;

