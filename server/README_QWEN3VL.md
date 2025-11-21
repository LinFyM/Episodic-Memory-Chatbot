# Qwen3-VL 多模态API服务器使用说明

## 概述

这是一个专门用于测试和验证文字+图片信息传递到Qwen3-VL大模型的API服务器。它支持：
- 接收QQ机器人客户端发送的文字和图片信息
- 使用Qwen3-VL模型进行多模态推理
- 返回生成结果给客户端

## 架构说明

```
QQ消息（文字+图片）
    ↓
ncatbot客户端 (qqbot_client_full.py)
    ↓ 提取图片，编码为base64
    ↓ HTTP POST
服务器API (api_server_qwen3vl.py)
    ↓ 转换为Qwen3-VL格式
    ↓ processor.apply_chat_template()
Qwen3-VL模型
    ↓ 生成回复
服务器API
    ↓ HTTP响应
客户端
    ↓ 发送回复
QQ用户
```

## 数据流程

### 1. 客户端发送的数据格式

**群聊消息：**
```json
{
    "type": "group",
    "group_id": "123456789",
    "group_name": "测试群",
    "user_id": "987654321",
    "user_nickname": "张三",
    "user_card": "张三",
    "content": "这是一条消息",
    "images": [
        {
            "data": "base64编码的图片数据...",
            "format": "jpeg"
        }
    ],
    "timestamp": 1234567890.0
}
```

**私聊消息：**
```json
{
    "type": "private",
    "user_id": "987654321",
    "user_nickname": "张三",
    "content": "这是一条消息",
    "images": [
        {
            "data": "base64编码的图片数据...",
            "format": "jpeg"
        }
    ],
    "timestamp": 1234567890.0
}
```

### 2. 服务器端处理流程

1. **接收请求**：从客户端接收JSON数据
2. **提取信息**：提取文字、图片（base64）、用户信息等
3. **格式化消息**：将消息格式化为Qwen3-VL需要的格式
   - 文本：`{"type": "text", "text": "消息内容"}`
   - 图片：`{"type": "image", "image": "data:image/jpeg;base64,..."}`
4. **构建聊天历史**：将消息添加到聊天历史中
5. **调用模型**：
   ```python
   inputs = processor.apply_chat_template(
       chat_history,
       tokenize=True,
       add_generation_prompt=True,
       return_dict=True,
       return_tensors="pt"
   )
   generated_ids = model.generate(**inputs, max_new_tokens=1000)
   ```
6. **解码回复**：将生成的token解码为文本
7. **返回响应**：返回JSON格式的回复

### 3. 服务器返回的数据格式

**群聊回复：**
```json
{
    "status": "success",
    "should_reply": true,
    "reply": "这是模型的回复内容"
}
```

**私聊回复：**
```json
{
    "status": "success",
    "reply": "这是模型的回复内容"
}
```

## 使用方法

### 1. 启动服务器

```bash
cd /data0/user/ymdai/LLM_memory/qqbot_new/server

# 使用默认配置
python api_server_qwen3vl.py

# 指定模型路径和设备
python api_server_qwen3vl.py \
    --model-path ./models/Qwen3-VL-8B-Instruct \
    --device cuda:0 \
    --host 0.0.0.0 \
    --port 9999
```

### 2. 配置客户端

编辑 `client/qqbot_client_full.py`，确保：
- `SERVER_URL` 指向服务器地址（如 `http://localhost:9999`）
- 客户端已正确配置QQ号和token

### 3. 启动客户端

```bash
cd /data0/user/ymdai/LLM_memory/qqbot_new/client
python qqbot_client_full.py
```

## 接口说明

### 健康检查

**GET** `/health`

返回：
```json
{
    "status": "healthy",
    "model_loaded": true,
    "processor_loaded": true,
    "device": "cuda:0"
}
```

### 群消息处理

**POST** `/api/chat/group`

请求体：见上面的"客户端发送的数据格式"

### 私聊消息处理

**POST** `/api/chat/private`

请求体：见上面的"客户端发送的数据格式"

## 关键技术点

### 1. 图片格式转换

客户端发送的图片是base64编码的字符串，服务器需要转换为Qwen3-VL支持的data URI格式：

```python
img_data_uri = f"data:image/{img_format};base64,{img_base64}"
```

### 2. 多模态消息格式

Qwen3-VL要求消息内容是一个列表，包含文本和图片：

```python
message_content = [
    {"type": "text", "text": "这是文本"},
    {"type": "image", "image": "data:image/jpeg;base64,..."},
    {"type": "image", "image": "data:image/png;base64,..."}
]
```

### 3. 使用processor.apply_chat_template()

这是Qwen3-VL处理多模态消息的关键API：

```python
inputs = processor.apply_chat_template(
    messages,  # 聊天历史列表
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt"
)
```

### 4. 模型生成参数

```python
generated_ids = model.generate(
    **inputs,
    max_new_tokens=1000,  # 最大生成token数
    temperature=0.7,      # 温度参数（控制随机性）
    do_sample=True,       # 是否采样
    pad_token_id=processor.tokenizer.eos_token_id
)
```

## 测试方法

### 1. 测试纯文本消息

在QQ群或私聊中发送纯文本消息，观察服务器日志输出。

### 2. 测试图片消息

在QQ群或私聊中发送图片（带或不带文字），观察：
- 客户端是否成功提取图片
- 服务器是否成功接收图片数据
- 模型是否能够理解图片内容并生成回复

### 3. 测试多图片消息

发送包含多张图片的消息，验证多模态处理是否正常。

## 故障排查

### 1. 模型加载失败

- 检查模型路径是否正确
- 检查设备是否可用（`nvidia-smi`查看GPU）
- 检查模型文件是否完整

### 2. 图片处理失败

- 检查图片base64编码是否正确
- 检查图片格式是否支持（jpeg, png, gif, webp）
- 查看服务器日志中的错误信息

### 3. 生成失败

- 检查内存是否充足
- 检查输入长度是否超过模型限制
- 查看服务器日志中的详细错误

### 4. 客户端连接失败

- 检查服务器是否正常启动
- 检查`SERVER_URL`配置是否正确
- 检查网络连接和防火墙设置

## 性能优化建议

1. **使用flash_attention_2**：在多图片场景下可以显著提升速度和节省内存
2. **批量处理**：如果有多个消息，可以考虑批量处理
3. **缓存机制**：对于相同的图片，可以缓存处理结果
4. **异步处理**：对于长文本生成，可以考虑异步处理

## 注意事项

1. **内存占用**：Qwen3-VL模型较大，需要足够的GPU内存
2. **图片大小**：建议限制图片大小，避免base64编码后数据过大
3. **生成速度**：多模态生成比纯文本慢，需要合理设置超时时间
4. **并发处理**：当前实现是单线程处理，如需并发需要额外优化

