# 记忆回忆机制 - 逐Token处理流程详解

## 概述

本文档详细说明自回归生成过程中，记忆回忆机制是如何在每个token的处理步骤中工作的。

---

## 一、生成循环结构

生成循环的核心结构（`custom_generate`函数）：

```python
while cur_len < max_new_tokens:
    1. 准备模型输入（prepare_inputs_for_generation）
    2. 检测<recall> token（触发回忆机制）
    3. 前向传播（model forward）
    4. 处理回忆机制（如果触发）
    5. 获取logits并采样下一个token
    6. 更新input_ids和model_kwargs
    7. 继续循环
```

---

## 二、详细Token处理流程

### 场景：输入包含"让我回忆一下<recall>"

假设输入序列为：`["让", "我", "回", "忆", "一", "下", "<recall>"]`

#### **循环第1轮：处理输入token "让"**

**步骤1：准备模型输入**
- `input_ids = [让, 我, 回, 忆, 一, 下, <recall>]`（完整输入）
- `model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)`
- 首次循环，`model_inputs['input_ids']` = 完整输入序列

**步骤2：检测<recall> token**
- `current_input_ids = model_inputs.get('input_ids')` = `[让, 我, 回, 忆, 一, 下, <recall>]`
- `last_token_id = current_input_ids[0, -1]` = `<recall>` token ID
- ✅ 检测到最后一个token是`<recall>`
- ✅ 设置 `recall_pending = True`

**步骤3：前向传播**
```python
outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
```
- 输入：`[让, 我, 回, 忆, 一, 下, <recall>]`
- 输出：`outputs.logits` shape = `[batch_size, seq_len, vocab_size]`
- `last_hidden_state = outputs.hidden_states[-1]` shape = `[batch_size, seq_len, hidden_size]`
- 最后一个位置的hidden state：`last_hidden_state[0, -1, :]` = `<recall>` token的隐藏向量

**步骤4：处理回忆机制**
- ✅ 检测到 `recall_pending = True`
- 提取查询向量：`query_vector = last_hidden_state[0, -1, :]`（`<recall>` token的hidden state）
- 从记忆库搜索：`memory_embedding, selected_meta = _sample_memory_embedding_from_db(query_vector)`
  - 使用余弦相似度搜索top_k个候选记忆
  - 使用temperature和top_p进行采样
  - 返回选中的记忆向量embedding
- 注入记忆向量：`memory_outputs = _inject_memory_embedding(memory_embedding)`
  ```python
  # 将记忆向量作为inputs_embeds注入
  memory_outputs = model(
      inputs_embeds=memory_embed,  # [1, 1, embed_dim]
      attention_mask=torch.ones(1, 1),
      past_key_values=model_kwargs.get('past_key_values'),  # 复用之前的KV cache
      use_cache=True,
      return_dict=True,
      output_hidden_states=True
  )
  ```
  - **关键**：使用`past_key_values`复用之前所有token（包括`<recall>`）的KV cache
  - 记忆向量作为**新的token**注入，位置在`<recall>`之后
  - 更新`model_kwargs['attention_mask']`：从`[1,1,1,1,1,1,1]`变为`[1,1,1,1,1,1,1,1]`（增加1位）
  - 更新`model_kwargs['past_key_values']`：包含所有8个位置的KV cache
- ✅ 替换outputs：`outputs = memory_outputs`
- ✅ 记录插入位置：`memory_injection_positions.append((input_ids.shape[-1], memory_score))`
  - 插入位置 = `input_ids.shape[-1]` = 7（`<recall>`的位置）

**步骤5：获取logits并采样**
- `next_token_logits = outputs.logits[:, -1, :]` 
  - 取最后一个位置的logits（记忆向量位置的logits）
- 应用LogitsProcessor和LogitsWarper
- 采样下一个token：`next_tokens = sample(next_token_logits)`
  - 例如：采样到token "用"

**步骤6：更新状态**
- `input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)`
  - `input_ids` = `[让, 我, 回, 忆, 一, 下, <recall>, 用]`
- 更新`model_kwargs`（包括past_key_values、attention_mask等）
- `cur_len += 1`

**步骤7：继续循环**

---

#### **循环第2轮：处理新生成的token "用"**

**步骤1：准备模型输入**
- `input_ids = [让, 我, 回, 忆, 一, 下, <recall>, 用]`
- `model_inputs = prepare_inputs_for_generation(input_ids, **model_kwargs)`
- 由于使用了KV cache，`model_inputs['input_ids']` = `[用]`（只传入新token）

**步骤2：检测<recall> token**
- `current_input_ids = [用]`
- `last_token_id = 用` token ID
- ❌ 不是`<recall>` token，不触发回忆

**步骤3：前向传播**
```python
outputs = model(**model_inputs, return_dict=True, output_hidden_states=True)
```
- 输入：`[用]`（只传入新token，KV cache包含之前所有token）
- 输出：`outputs.logits` shape = `[batch_size, 1, vocab_size]`（只有新token的logits）

**步骤4：处理回忆机制**
- ❌ `recall_pending = False`，跳过回忆处理

**步骤5：获取logits并采样**
- `next_token_logits = outputs.logits[:, -1, :]`（"用"的logits）
- 采样下一个token：例如 "户"

**步骤6：更新状态**
- `input_ids = [让, 我, 回, 忆, 一, 下, <recall>, 用, 户]`
- 更新`model_kwargs`
- `cur_len += 1`

**步骤7：继续循环**

---

## 三、关键机制说明

### 3.1 记忆向量注入的时机

**触发条件**：
1. 检测到最后一个输入token是`<recall>` token
2. 设置`recall_pending = True`
3. 前向传播后，提取`<recall>` token的hidden state作为查询向量
4. 从记忆库搜索匹配的记忆向量
5. 将记忆向量作为`inputs_embeds`注入模型

**注入方式**：
```python
memory_outputs = model(
    inputs_embeds=memory_embed,  # [1, 1, embed_dim] - 记忆向量
    past_key_values=model_kwargs.get('past_key_values'),  # 复用之前的KV cache
    ...
)
```

**关键点**：
- 记忆向量**不是**作为token ID注入，而是作为embedding直接注入
- 复用之前所有token的KV cache，保持上下文连续性
- 记忆向量注入后，模型会基于记忆向量生成下一个token

### 3.2 KV Cache的复用

**首次循环（处理输入序列）**：
- 输入：`[让, 我, 回, 忆, 一, 下, <recall>]`
- 生成KV cache：`past_key_values`包含7个位置的KV

**记忆向量注入**：
- 输入：记忆向量embedding（作为`inputs_embeds`）
- 复用KV cache：`past_key_values`（包含之前7个位置的KV）
- 生成新的KV：记忆向量位置的KV
- 更新KV cache：`past_key_values`包含8个位置的KV

**后续循环**：
- 输入：新生成的token（如"用"）
- 复用KV cache：`past_key_values`（包含之前8个位置的KV）
- 生成新的KV：新token位置的KV
- 更新KV cache：`past_key_values`包含9个位置的KV

### 3.3 记忆向量插入位置的记录

**记录时机**：
- 在记忆向量注入成功后记录
- `injection_pos = input_ids.shape[-1]`
- 此时`input_ids`还**没有**包含记忆向量后的新token

**记录内容**：
- `(token_position, memory_score)`
- `token_position`：记忆向量插入的绝对位置（相对于完整序列）
- `memory_score`：记忆向量匹配的相似度分数

**示例**：
- 输入序列长度：7（`[让, 我, 回, 忆, 一, 下, <recall>]`）
- 记忆向量插入位置：7（在`<recall>`之后）
- 记录：`(7, 0.8438)`

### 3.4 下一个token的生成

**记忆向量注入后**：
- `outputs = memory_outputs`（记忆向量注入后的outputs）
- `next_token_logits = outputs.logits[:, -1, :]`（记忆向量位置的logits）
- 基于记忆向量生成下一个token

**关键点**：
- 下一个token是基于记忆向量的语义生成的
- 不是直接输出记忆文本，而是基于记忆向量生成相关内容

---

## 四、完整流程示例

### 输入：`"让我回忆一下<recall>"`

**Token序列**：
```
输入: [让, 我, 回, 忆, 一, 下, <recall>]
```

**循环1：处理输入序列**
1. 检测到`<recall>` token
2. 前向传播，提取`<recall>`的hidden state
3. 搜索记忆库，找到匹配的记忆向量（相似度0.8438）
4. 注入记忆向量（作为embedding）
5. 基于记忆向量生成下一个token："用"
6. `input_ids` = `[让, 我, 回, 忆, 一, 下, <recall>, 用]`

**循环2：处理新token "用"**
1. 前向传播（复用KV cache）
2. 生成下一个token："户"
3. `input_ids` = `[让, 我, 回, 忆, 一, 下, <recall>, 用, 户]`

**循环3：处理新token "户"**
1. 前向传播（复用KV cache）
2. 生成下一个token："的"
3. `input_ids` = `[让, 我, 回, 忆, 一, 下, <recall>, 用, 户, 的]`

**...继续生成...**

**最终输出**：
```
让我回忆一下<recall>用户的生日是1990年1月1日...
```

**记忆插入位置记录**：
- 位置7：在`<recall>`之后插入了记忆向量（相似度0.8438）

---

## 五、关键代码位置

### 5.1 检测<recall> token
```python
# 第2403-2418行
current_input_ids = model_inputs.get('input_ids', input_ids)
if current_input_ids.shape[-1] > 0:
    last_token_id = current_input_ids[0, -1].item()
    if last_token_id == recall_token_id and not recall_pending:
        recall_pending = True
```

### 5.2 提取查询向量
```python
# 第2432行
query_vector = last_hidden_state[0, -1, :]  # <recall> token的hidden state
```

### 5.3 记忆向量注入
```python
# 第2361-2368行
memory_outputs = model(
    inputs_embeds=memory_embed,  # [1, 1, embed_dim]
    attention_mask=attention_mask,
    past_key_values=model_kwargs.get('past_key_values'),  # 复用KV cache
    use_cache=True,
    return_dict=True,
    output_hidden_states=True
)
```

### 5.4 记录插入位置
```python
# 第2449-2450行
injection_pos = input_ids.shape[-1]  # 记忆向量插入在当前位置之后
memory_injection_positions.append((injection_pos, memory_score))
```

### 5.5 基于记忆向量生成下一个token
```python
# 第2445行
outputs = memory_outputs  # 使用记忆向量注入后的outputs
# 第2455行
next_token_logits = outputs.logits[:, -1, :]  # 记忆向量位置的logits
```

---

## 六、总结

### 记忆回忆机制的核心流程：

1. **检测触发**：检测到最后一个输入token是`<recall>`
2. **提取查询**：前向传播后提取`<recall>` token的hidden state
3. **搜索记忆**：使用查询向量从记忆库搜索匹配的记忆向量
4. **注入记忆**：将记忆向量作为embedding注入模型（复用KV cache）
5. **生成内容**：基于记忆向量生成下一个token
6. **继续生成**：后续token基于记忆向量和上下文继续生成

### 关键特点：

- ✅ 记忆向量**不是**作为token ID，而是作为embedding直接注入
- ✅ 复用之前所有token的KV cache，保持上下文连续性
- ✅ 记忆向量注入后立即完成，无需额外的状态管理
- ✅ 下一个token基于记忆向量的语义生成，不是直接输出记忆文本

