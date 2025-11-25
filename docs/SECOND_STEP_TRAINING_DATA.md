# 第二步训练数据构造详解

本文档详细说明第二步训练中三种训练数据的具体构造方式以及label设置。

## 三种训练数据类型

1. **记忆-前置（memory_front）**：记忆条目 + 随机SFT前缀
2. **记忆-前后拼接（memory_full）**：记忆条目 + SFT前缀 + SFT后缀（夹心结构）
3. **纯SFT（sft_only）**：纯SFT样本，只训练assistant输出

---

## 1. 记忆-前置（memory_front）

### 数据构造

**位置**：`src/training/text_memory_train.py::_get_memory_decode_sample()`

**完整序列结构**：
```
[context_text] + [activation_prompt] + <recall> + <|memory_pad|> + [memory_text] + </recall> + [end_prompt]
```

**详细说明**：
- **`context_text`**：随机SFT前缀（从 `sft_full_texts` 中抽取，截断到思考部分之前），如果没有SFT数据则从其他记忆条目中随机选择
- **`activation_prompt`**：随机选择的激活引导语（如"（让我切换到回忆模式……）"）
- **`<recall>`**：回忆开始标记
- **`<|memory_pad|>`**：记忆向量插入位置（会被记忆向量替换）
- **`memory_text`**：记忆条目的实际文本内容
- **`</recall>`**：回忆结束标记
- **`end_prompt`**：随机选择的结束引导语（如"——回忆完成。"）

### Label设置

**位置**：`src/training/text_memory_train.py::_get_memory_decode_sample()` (第368-386行)

**Label规则**：
```python
# 前缀部分（context + activation + recall）：全部为-100（不训练）
prefix_labels = [-100] * prefix_len

# 但是<recall> token本身需要训练（设置为实际的token_id）
for offset, token_id in enumerate(recall_tokens):
    pos = recall_start_idx + offset
    if 0 <= pos < prefix_len:
        prefix_labels[pos] = token_id  # <recall> token有label

# 目标部分（memory_text + </recall> + end_prompt）：全部有label
target_tokens = actual_tokenizer(target_text, add_special_tokens=False)['input_ids']
# target_text = memory_text + </recall> + end_prompt

# 最终labels
labels = prefix_labels + target_tokens
```

**具体Label分布**：
```
位置范围                    Label值
─────────────────────────────────────────────
[0, context_len)            -100（不训练）
[context_len, activation_end) -100（不训练）
<recall> token位置          token_id（训练）
<|memory_pad|>位置          -100（不训练，会被向量替换）
[memory_start, memory_end)  token_id（训练记忆文本）
</recall>位置                token_id（训练）
[end_prompt位置]            token_id（训练）
```

**示例**（假设context有10个token，activation有5个token，memory有20个token）：
```
Input:  [ctx1, ctx2, ..., ctx10, act1, ..., act5, <recall>, <|memory_pad|>, mem1, ..., mem20, </recall>, end1, end2]
Labels: [-100, -100, ..., -100, -100, ..., -100, recall_id, -100, mem1_id, ..., mem20_id, recall_end_id, end1_id, end2_id]
         ↑───────────不训练──────────↑  ↑训练↑  ↑不训练↑  ↑────────训练────────↑
```

---

## 2. 记忆-前后拼接（memory_full）

### 数据构造

**位置**：`src/training/text_memory_train.py::_get_memory_decode_sample()` (context_override参数)

**完整序列结构**：
```
[prefix_text] + [activation_prompt] + <recall> + <|memory_pad|> + [memory_text] + </recall> + [end_prompt] + [suffix_text]
```

**详细说明**：
- **`prefix_text`**：SFT完整文本的前半部分（从开始到`<think>`标签之前）
- **`activation_prompt`**：随机选择的激活引导语
- **`<recall>`**：回忆开始标记
- **`<|memory_pad|>`**：记忆向量插入位置
- **`memory_text`**：记忆条目的实际文本内容
- **`</recall>`**：回忆结束标记
- **`end_prompt`**：随机选择的结束引导语
- **`suffix_text`**：SFT完整文本的后半部分（从`</think>`标签之后到结束）

**关键点**：
- `prefix_text` 和 `suffix_text` 来自**同一个SFT样本**，在`<think>`标签处被截断
- 记忆条目被"夹"在SFT样本的中间，形成夹心结构

### Label设置

**Label规则**（与memory_front类似，但增加了suffix部分）：
```python
# 前缀部分（prefix + activation + recall）：全部为-100（不训练）
prefix_labels = [-100] * prefix_len

# <recall> token本身需要训练
prefix_labels[recall_start_idx:recall_start_idx + recall_token_count] = recall_tokens

# 目标部分（memory_text + </recall> + end_prompt + suffix_text）：全部有label
target_text = f"{memory_text}{</recall>}{end_prompt}{suffix_text}"
target_tokens = actual_tokenizer(target_text, add_special_tokens=False)['input_ids']

# 最终labels
labels = prefix_labels + target_tokens
```

**具体Label分布**：
```
位置范围                    Label值
─────────────────────────────────────────────
[0, prefix_len)             -100（不训练）
[prefix_len, activation_end) -100（不训练）
<recall> token位置          token_id（训练）
<|memory_pad|>位置          -100（不训练）
[memory_start, memory_end)  token_id（训练记忆文本）
</recall>位置                token_id（训练）
[end_prompt位置]            token_id（训练）
[suffix_start, suffix_end)  token_id（训练SFT后缀）
```

**示例**（假设prefix有15个token，activation有5个token，memory有20个token，suffix有25个token）：
```
Input:  [pre1, ..., pre15, act1, ..., act5, <recall>, <|memory_pad|>, mem1, ..., mem20, </recall>, end1, end2, suf1, ..., suf25]
Labels: [-100, ..., -100, -100, ..., -100, recall_id, -100, mem1_id, ..., mem20_id, recall_end_id, end1_id, end2_id, suf1_id, ..., suf25_id]
         ↑────────不训练────────↑              ↑训练↑  ↑不训练↑  ↑────────训练────────↑              ↑────训练────↑
```

---

## 3. 纯SFT（sft_only）

### 数据构造

**位置**：`src/training/text_memory_train.py::_get_sft_sample()`

**完整序列结构**：
```
完整的SFT messages（经过apply_chat_template转换）
```

**详细说明**：
- 使用 `tokenizer.apply_chat_template()` 将messages转换为完整的对话序列
- 包含system、user、assistant等所有角色的消息
- 不包含任何记忆相关的特殊token（`<recall>`、`<|memory_pad|>`等）

### Label设置

**位置**：`src/training/text_memory_train.py::_get_sft_sample()` (第769-797行)

**Label规则**：
```python
# 默认全部mask
labels_tensor = torch.full_like(input_ids, -100)

# 计算每条message结束时的长度
prefix_lengths = []
for end_idx in range(len(messages)):
    prefix_slice = messages[: end_idx + 1]
    prefix_ids = tokenizer.apply_chat_template(prefix_slice, ...)['input_ids'][0]
    prefix_lengths.append(prefix_ids.shape[0])

# 只对assistant角色的内容设置label
prev_len = 0
for msg_idx, message in enumerate(messages):
    curr_len = prefix_lengths[msg_idx]
    if message.get("role") == "assistant":
        labels_tensor[prev_len:curr_len] = input_ids[prev_len:curr_len]  # 训练assistant输出
    prev_len = curr_len
```

**具体Label分布**：
```
位置范围                    Label值
─────────────────────────────────────────────
[0, system_end)             -100（不训练）
[system_end, user1_end)     -100（不训练）
[user1_end, assistant1_end) token_id（训练assistant输出）
[assistant1_end, user2_end) -100（不训练）
[user2_end, assistant2_end) token_id（训练assistant输出）
...
```

**示例**（假设有system、user、assistant各一条消息）：
```
Input:  [sys1, ..., sys5, user1, ..., user10, ass1, ..., ass20]
Labels: [-100, ..., -100, -100, ..., -100, ass1_id, ..., ass20_id]
         ↑────不训练────↑  ↑────不训练────↑  ↑────训练────↑
```

---

## Collate函数处理

**位置**：`src/training/text_memory_train.py::enhanced_collate_fn()`

### 记忆条目样本处理

```python
# 将input_tokens和target_labels拼接
target_labels = item_labels[len(input_tokens):]
total_tokens = torch.cat([input_tokens, target_labels])
total_len = len(total_tokens)

# 填充到batch
input_ids[i, :total_len] = total_tokens
attention_mask[i, :total_len] = 1
labels[i, :len(item_labels)] = item_labels  # 直接使用预先计算好的labels
```

### SFT样本处理

```python
# SFT样本：input_ids和labels长度相同
total_len = len(input_tokens)
input_ids[i, :total_len] = input_tokens
attention_mask[i, :total_len] = 1
labels[i, :len(item_labels)] = item_labels  # 直接使用预先计算好的labels
```

---

## 总结对比

| 数据类型 | 输入结构 | Label训练部分 | 特殊处理 |
|---------|---------|--------------|---------|
| **memory_front** | context + activation + `<recall>` + `<|memory_pad|>` + memory + `</recall>` + end | memory + `</recall>` + end | `<recall>` token有label，`<|memory_pad|>`位置会被向量替换 |
| **memory_full** | prefix + activation + `<recall>` + `<|memory_pad|>` + memory + `</recall>` + end + suffix | memory + `</recall>` + end + suffix | 同memory_front，但增加了suffix训练 |
| **sft_only** | 完整SFT messages | 仅assistant输出 | 不包含任何记忆相关token |

## 关键设计原则

1. **记忆条目训练**：
   - 前缀部分（context/prefix + activation）不训练，让模型学会忽略干扰
   - 只有记忆内容本身和结束标记被训练
   - `<recall>` token需要训练，确保模型能正确生成

2. **SFT训练**：
   - 只训练assistant的输出，不训练输入部分
   - 保持标准的对话微调格式

3. **夹心结构**：
   - 记忆条目被"夹"在SFT样本中间，训练模型在回忆后继续正常输出
   - suffix部分也被训练，确保模型能完整输出SFT样本的后续内容

