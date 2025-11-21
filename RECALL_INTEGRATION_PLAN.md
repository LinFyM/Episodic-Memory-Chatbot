# 回忆机制集成计划

## 一、核心组件集成

### 1.1 MemoryVectorDB 集成
- 从 `recall/chat_with_recall.py` 导入 `MemoryVectorDB` 类
- 在服务器启动时初始化 `MemoryVectorDB`
- 加载记忆数据（.pt文件）

### 1.2 特殊Token检查
- 检查模型是否包含特殊token：`<recall>`, `<|recall|>`, `</recall>`
- 如果不存在，需要先训练模型或使用已训练的模型

## 二、生成流程改造

### 2.1 修改 `custom_generate` 函数
需要在生成循环中检测 `<recall>` token，并触发回忆流程：

1. **检测 `<recall>` token**
   - 在生成循环中，每次生成新token后检查是否为 `<recall>`
   - 如果是，进入回忆模式

2. **获取查询向量**
   - 将 `<recall>` token输入模型（使用KV缓存）
   - 获取最后一层的hidden states作为查询向量

3. **记忆检索**
   - 使用查询向量在 `MemoryVectorDB` 中搜索
   - 获取 top_k 个最相似的记忆
   - 选择最匹配的记忆（相似度最高）

4. **输入记忆embedding**
   - 将记忆的embedding通过 `inputs_embeds` 输入模型
   - 更新KV缓存和attention_mask

5. **生成 `<|recall|>` token**
   - 输入 `<|recall|>` token
   - 更新KV缓存
   - 继续正常生成流程

### 2.2 支持中断机制
- 回忆过程也需要支持中断
- 在记忆检索和embedding输入过程中检查 `interrupt_event`

## 三、配置文件更新

### 3.1 添加回忆相关配置
```yaml
recall:
  enabled: true  # 是否启用回忆功能
  memory_path: "./memory_data/text_embeddings.pt"  # 记忆数据路径
  top_k: 5  # 检索top_k个记忆
  similarity_threshold: 0.5  # 相似度阈值（可选）
```

## 四、实现步骤

### 步骤1：导入必要组件
- 从 `recall/chat_with_recall.py` 导入 `MemoryVectorDB`
- 在服务器初始化时加载

### 步骤2：修改生成函数
- 在 `custom_generate` 中添加回忆流程
- 保持与现有中断机制兼容

### 步骤3：添加配置支持
- 在配置文件中添加回忆相关配置
- 支持动态启用/禁用回忆功能

### 步骤4：测试和验证
- 测试回忆功能是否正常工作
- 验证中断机制是否兼容
- 检查性能影响

## 五、注意事项

1. **设备一致性**：确保MemoryVectorDB与模型在同一设备上
2. **数据类型**：确保embedding的数据类型与模型匹配（通常是bfloat16）
3. **KV缓存**：回忆过程中需要正确更新KV缓存
4. **中断机制**：回忆流程的每一步都需要检查中断信号
5. **性能优化**：记忆检索可能较慢，需要考虑异步或缓存

## 六、代码结构

```
api_server_qwen3vl.py
├── 导入MemoryVectorDB
├── 初始化memory_db
├── 加载记忆数据
├── custom_generate函数
│   ├── 生成循环
│   ├── 检测<recall> token
│   ├── 获取查询向量
│   ├── 记忆检索
│   ├── 输入记忆embedding
│   └── 生成<|recall|> token
└── 配置支持
```

