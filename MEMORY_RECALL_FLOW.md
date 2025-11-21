# 记忆回忆激活流程详解

## 概述

本文档详细说明整个记忆回忆激活机制的工作流程。该机制允许模型在生成过程中自动检测`<recall>` token，从记忆库中检索相关记忆向量，并将其注入到生成过程中。

---

## 一、初始化阶段

### 1.1 配置加载
- 从`config_qwen3vl.yaml`加载记忆相关配置：
  - `memory.autoregressive_recall.enabled`: 是否启用自动回忆
  - `memory.autoregressive_recall.top_k`: 检索top_k个候选记忆
  - `memory.autoregressive_recall.top_p`: top_p截断阈值
  - `memory.autoregressive_recall.temperature`: 采样温度
  - `memory.autoregressive_recall.debug`: 是否开启调试日志

### 1.2 记忆库初始化
- 创建`MemoryVectorDB`实例
- 从`.pt`文件加载记忆向量（只包含embedding，不包含文本）
- 记忆向量已归一化，用于余弦相似度计算

### 1.3 Token ID获取
- 从`recall_token_ids`全局变量获取：
  - `<recall>` token ID（用于触发回忆机制）

---

## 二、生成循环开始

### 2.1 输入准备
- `custom_generate`函数接收`inputs`字典（包含`input_ids`和`attention_mask`）
- 初始化状态变量：
  - `cur_len = 0`: 当前生成长度
  - `recall_pending = False`: 是否待触发回忆（避免重复触发）
  - `memory_injection_positions = []`: 记录记忆插入位置

### 2.2 KV Cache初始化
- 调用`model.prepare_inputs_for_generation()`准备首次前向传播的输入
- 初始化`cache_position`（用于KV cache的位置索引）
- 设置`use_cache=True`以启用KV cache加速

---

## 三、生成循环（核心流程）

### 3.1 循环开始
```python
while cur_len < max_new_tokens:
```

### 3.2 准备模型输入
- 调用`model.prepare_inputs_for_generation(input_ids, **model_kwargs)`
- 该方法会自动处理：
  - **KV cache裁剪**：如果存在KV cache，只传入未缓存的token
  - **attention_mask更新**：根据当前序列长度更新
  - **position_ids处理**：设置正确的位置编码
  - **cache_position更新**：更新缓存位置索引

### 3.3 🔍 **检测`<recall>` token（关键步骤）**

在每次前向传播**之前**，检查当前要处理的最后一个token：

```python
current_input_ids = model_inputs.get('input_ids', input_ids)
if current_input_ids.shape[-1] > 0:
    last_token_id = current_input_ids[0, -1].item()
    if (
        autorecall_enabled
        and recall_token_id is not None
        and last_token_id == recall_token_id
        and not recall_pending  # 避免重复触发
    ):
        # 检查记忆库
        if memory_db is not None and len(memory_db) > 0:
            recall_pending = True  # 设置待触发标志
```

**触发条件**：
1. ✅ 自动回忆功能已启用
2. ✅ `recall_token_id`不为None
3. ✅ 当前最后一个token是`<recall>` token
4. ✅ 未处于待触发状态（避免重复触发）
5. ✅ 记忆库不为空

**触发时机**：
- **初始输入**：如果用户输入末尾是`<recall>` token，第一次循环就会触发
- **自回归生成**：如果模型生成了`<recall>` token，下一轮循环的前向传播前会检测到

### 3.4 前向传播
```python
outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
last_hidden_state = outputs.hidden_states[-1]  # [batch_size, seq_len, hidden_size]
```

- 获取模型的logits和hidden states
- `last_hidden_state`用于提取查询向量

### 3.5 🔄 **回忆机制触发（核心逻辑）**

如果`recall_pending == True`：

#### 步骤1：提取查询向量
```python
query_vector = last_hidden_state[0, -1, :]  # [hidden_size]
```
- 从`<recall>` token位置的hidden state提取查询向量
- shape: `[hidden_size]`（例如`[2560]`）

#### 步骤2：向量匹配（`_sample_memory_embedding_from_db`）

**2.1 搜索记忆库**
```python
search_results = memory_db.search(
    query_vec.detach().clone(),
    top_k=max(autorecall_top_k, 1),
    debug=autorecall_debug
)
```

**搜索过程**：
1. 归一化查询向量：`query_normalized = F.normalize(query_vec, p=2, dim=-1)`
2. 计算余弦相似度：`similarities = query_normalized @ memory_embeddings.T`
3. 获取top_k个最相似的结果
4. 返回结果列表，每个结果包含：
   - `score`: 相似度分数（0-1之间）
   - `embedding`: 记忆向量tensor
   - `index`: 在记忆库中的索引

**2.2 Top-p截断（可选）**
```python
if 0 < autorecall_top_p < 1.0:
    # 按相似度排序
    sorted_scores, sorted_indices = torch.sort(scores, descending=True)
    # 计算概率分布
    probs = torch.softmax(sorted_scores / temperature, dim=-1)
    # 累积概率
    cumulative = torch.cumsum(probs, dim=-1)
    # 保留累积概率 <= top_p 的候选
    cutoff_mask = cumulative <= autorecall_top_p
    valid_indices = sorted_indices[cutoff_mask]
```

**2.3 采样选择**
```python
if do_sample:
    # 采样方式：根据概率分布随机选择
    probs = torch.softmax(scores / temperature, dim=-1)
    choice_idx = torch.multinomial(probs, num_samples=1).item()
else:
    # 贪婪方式：选择相似度最高的
    choice_idx = torch.argmax(scores).item()
```

**2.4 返回选中的记忆向量**
```python
selected = search_results[choice_idx]
embedding_tensor = selected['embedding']  # [embedding_dim]
return embedding_tensor, selected  # selected包含score等信息
```

#### 步骤3：向量注入（`_inject_memory_embedding`）

**3.1 准备记忆向量**
```python
memory_embed = memory_embedding_tensor.to(
    dtype=memory_dtype,  # 通常为bfloat16
    device=actual_device  # 模型所在设备
).unsqueeze(0).unsqueeze(0)  # [1, 1, embed_dim]
```

**3.2 构建attention mask**
```python
attention_mask = torch.ones(1, 1, device=device, dtype=torch.long)
```

**3.3 前向传播注入**
```python
memory_outputs = model(
    inputs_embeds=memory_embed,  # 使用embedding而不是token ID
    attention_mask=attention_mask,
    past_key_values=model_kwargs.get('past_key_values'),  # 使用现有KV cache
    use_cache=True,
    return_dict=True,
    output_hidden_states=True
)
```

**关键点**：
- 使用`inputs_embeds`而不是`input_ids`，直接将记忆向量作为输入
- 复用现有的`past_key_values`（KV cache），确保上下文连续性
- 记忆向量会经过模型的前向传播，生成对应的logits和hidden states

**3.4 更新model_kwargs**
```python
# 更新attention_mask
model_kwargs['attention_mask'] = torch.cat(
    [model_kwargs['attention_mask'], attention_mask],
    dim=1
)

# 更新past_key_values（KV cache）
_update_model_kwargs_helper(memory_outputs)
```

**3.5 记录插入位置**
```python
injection_pos = input_ids.shape[-1]  # 当前input_ids的长度
memory_score = selected_meta.get("score", 0.0)
memory_injection_positions.append((injection_pos, memory_score))
```

**3.6 更新状态**
```python
outputs = memory_outputs  # 使用记忆向量注入后的outputs
recall_pending = False  # 清除待触发标志
# 注意：插入完记忆向量后立即完成，无需额外的"回忆模式"状态
```

### 3.6 生成下一个token

**3.6.1 获取logits**
```python
next_token_logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
```

**3.6.2 应用LogitsProcessor**
```python
next_token_scores = logits_processor(input_ids, next_token_logits)
```
- 例如：`RepetitionPenaltyLogitsProcessor`用于惩罚重复token

**3.6.3 应用LogitsWarper**
```python
if do_sample and logits_warper is not None:
    next_token_scores = logits_warper(input_ids, next_token_scores)
```
- `TemperatureLogitsWarper`: 温度调节
- `TopKLogitsWarper`: top-k截断
- `TopPLogitsWarper`: top-p截断

**3.6.4 采样token**
```python
if do_sample:
    probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
    next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
else:
    next_tokens = torch.argmax(next_token_scores, dim=-1)
```

**3.6.5 注意：不需要检测`</recall>` token**
- 记忆向量插入完成后立即完成回忆流程
- `</recall>` token按普通token处理（如果模型生成的话）

### 3.7 更新状态

**3.7.1 更新input_ids**
```python
input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
```

**3.7.2 更新model_kwargs**
```python
model_kwargs = model._update_model_kwargs_for_generation(
    outputs,
    model_kwargs,
    is_encoder_decoder=False,
    standardize_cache_format=True,
)
```
- 更新`past_key_values`（KV cache）
- 更新`attention_mask`
- 更新`cache_position`

**3.7.3 检查停止条件**
```python
# EOS token检查
if eos_token_ids is not None:
    eos_in_sentence = (next_tokens.unsqueeze(-1) == eos_token_ids.unsqueeze(0)).any(dim=-1)
    unfinished_sequences = unfinished_sequences & ~eos_in_sentence

# StoppingCriteria检查
should_stop = stopping_criteria(input_ids, next_token_scores)
```

**3.7.4 更新循环计数**
```python
cur_len += 1
```

---

## 四、生成完成

### 4.1 返回结果
```python
if memory_injection_positions:
    return input_ids, memory_injection_positions  # 包含插入位置信息
else:
    return input_ids  # 普通返回
```

### 4.2 位置标注
- 在生成结果中标注记忆向量插入位置
- 显示token位置和相似度分数

---

## 五、关键设计点

### 5.1 统一触发机制
- **之前**：检测到生成`<recall>` token后触发
- **现在**：检测到最新输入是`<recall>` token时触发
- **优势**：无论是初始输入还是自回归生成，都能统一处理

### 5.2 KV Cache复用
- 记忆向量注入时复用现有的KV cache
- 确保上下文连续性，不会丢失之前的生成历史

### 5.3 状态管理
- `recall_pending`标记待处理的回忆请求（避免重复触发）
- 记忆向量插入完成后立即完成，无需额外的状态管理

### 5.4 采样策略
- 支持贪婪和采样两种方式
- 使用temperature、top_k、top_p控制采样多样性

---

## 六、流程图

```
开始生成
  ↓
初始化状态变量
  ↓
┌─────────────────┐
│  生成循环开始   │
└─────────────────┘
  ↓
准备模型输入（prepare_inputs_for_generation）
  ↓
🔍 检测最后一个token是否是<recall>
  ├─ 是 → 设置recall_pending=True
  └─ 否 → 继续
  ↓
前向传播（获取hidden states）
  ↓
检查recall_pending
  ├─ True → 🔄 触发回忆机制
  │   ├─ 提取查询向量（<recall>的hidden state）
  │   ├─ 🔍 向量匹配
  │   │   ├─ 搜索记忆库（余弦相似度）
  │   │   ├─ Top-p截断（可选）
  │   │   └─ 采样选择
  │   ├─ 💉 向量注入
  │   │   ├─ 准备记忆向量
  │   │   ├─ 前向传播注入（复用KV cache）
  │   │   └─ 更新model_kwargs
  │   └─ 清除recall_pending标志
  └─ False → 继续
  ↓
获取logits（outputs.logits[:, -1, :]）
  ↓
应用LogitsProcessor和LogitsWarper
  ↓
采样下一个token
  ↓
更新input_ids和model_kwargs
  ↓
检查停止条件
  ├─ 满足 → 结束生成
  └─ 不满足 → 继续循环
  ↓
返回结果（包含记忆插入位置信息）
```

---

## 七、日志输出示例

```
🎯 [输入检测] 检测到最新输入是<recall> token (ID: 151669)，触发回忆机制
🔄 [回忆触发] 检测到recall_pending=True，开始处理回忆机制
🔍 [回忆触发] 提取<recall> token的hidden state作为查询向量，shape: torch.Size([2560])
🔍 [向量匹配] 开始搜索记忆库，查询向量shape: torch.Size([2560]), top_k=10
🔍 [向量匹配] 找到 8 个候选记忆向量
  [1] 相似度=0.8438
  [2] 相似度=0.8398
  ...
🔍 [向量匹配] top_p=0.95 截断后保留 7 个候选
🔍 [向量匹配] 使用采样方式选择记忆，选择索引: 0, 概率: 0.1444
✅ [向量匹配] 已选择记忆向量，相似度=0.8438
🎯 [回忆触发] 采样到记忆向量，相似度=0.8438
💉 [向量插入] 开始注入记忆向量，shape: torch.Size([1, 1, 2560]), device: cuda:0, dtype: torch.bfloat16
💉 [向量插入] 检测到已存在的KV cache，将在此基础上注入记忆向量
💉 [向量插入] 记忆向量前向传播完成，outputs.logits.shape: torch.Size([1, 1, 151671])
💉 [向量插入] 更新attention_mask: 6 -> 7
✅ [向量插入] 记忆向量注入成功，已更新model_kwargs
✅ [回忆触发] 记忆向量注入成功
📍 [位置记录] 记忆向量插入位置: token位置=6, 相似度=0.8438
```

---

## 八、注意事项

1. **记忆库结构**：只存储embedding向量，不存储文本，因此无法显示记忆文本预览
2. **KV Cache**：记忆向量注入时复用现有KV cache，确保上下文连续性
3. **位置记录**：记录的是token位置（`input_ids`的长度），不是文本位置
4. **立即完成**：记忆向量插入后立即完成，无需额外的状态管理
5. **采样策略**：支持贪婪和采样两种方式，可通过配置调整

