# 训练架构说明

## 目录结构

### 1. `server/memory/` - 训练流程编排层（Orchestration Layer）

**作用**：负责整个训练流程的编排、数据准备、模型管理

**文件**：
- `training_service.py` - **核心训练服务**，负责：
  - 保存和加载聊天记录
  - 提取记忆条目（`extract_memory_entries`）
  - 调用训练器进行训练（`train_recall_token`, `train_memory_decoding`）
  - 管理训练流程（`run_training`）
  - 保存训练好的模型
- `training_scheduler.py` - 训练调度器，负责定时触发训练
- `vector_db.py` - 向量数据库，存储记忆向量
- `token_manager.py` - Token管理器，管理特殊token

### 2. `recall/` - 训练器实现层（Trainer Implementation Layer）

**作用**：包含具体的训练器实现，执行实际的模型训练

**文件**：
- `text_embedding_train.py` - 包含 `RecallMemoryTrainer` 类
  - 负责第一步训练：训练 `<recall>` token 的 embedding
  - 使用 LoRA 进行训练
  - 支持 K-fold 交叉验证
- `text_memory_train.py` - 包含 `EnhancedTextMemoryTrainer` 类
  - 负责第二步训练：训练记忆解码能力
  - 训练模型从 `<recall>` token + 记忆向量 解码出记忆文本
  - 支持 SFT 数据注入
- `get_text_embedding.py` - 包含 `extract_last_token_embedding` 函数
  - 用于提取模型最后一个 token 的 embedding

## 训练流程

```
┌─────────────────────────────────────────────────────────────┐
│  server/memory/training_service.py                          │
│  MemoryTrainingService.run_training()                        │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  步骤0: 保存聊天记录到JSON        │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  步骤1: 加载JSON文件中的聊天记录  │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  步骤2: 提取记忆条目               │
        │  extract_memory_entries()          │
        │  (使用基础模型)                     │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  步骤3: 训练                       │
        │  ┌─────────────────────────────┐  │
        │  │ 第一步: train_recall_token() │  │
        │  │ 导入: RecallMemoryTrainer   │  │
        │  │ 来源: recall/text_embedding_ │  │
        │  │       train.py              │  │
        │  └─────────────────────────────┘  │
        │  ┌─────────────────────────────┐  │
        │  │ 第二步: train_memory_       │  │
        │  │        decoding()           │  │
        │  │ 导入: EnhancedTextMemory    │  │
        │  │       Trainer               │  │
        │  │ 来源: recall/text_memory_   │  │
        │  │       train.py              │  │
        │  └─────────────────────────────┘  │
        └───────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────┐
        │  步骤4: 保存训练好的模型           │
        └───────────────────────────────────┘
```

## 导入关系

```
server/memory/training_service.py
    │
    ├─→ 导入 recall/text_embedding_train.py
    │   └─→ RecallMemoryTrainer 类
    │
    ├─→ 导入 recall/text_memory_train.py
    │   └─→ EnhancedTextMemoryTrainer 类
    │
    └─→ 导入 recall/get_text_embedding.py
        └─→ extract_last_token_embedding 函数
```

## 关键代码位置

### 训练流程入口
- `server/memory/training_service.py::MemoryTrainingService.run_training()` (第591行)

### 第一步训练
- `server/memory/training_service.py::MemoryTrainingService.train_recall_token()` (第2042行)
- 导入：`from text_embedding_train import RecallMemoryTrainer` (第141行)
- 使用：`trainer = RecallMemoryTrainer(...)` (第2080行)

### 第二步训练
- `server/memory/training_service.py::MemoryTrainingService.train_memory_decoding()` (第2135行)
- 导入：`from text_memory_train import EnhancedTextMemoryTrainer` (第142行)
- 使用：`trainer = EnhancedTextMemoryTrainer(...)` (第2174行)

## 总结

**两个目录都是必需的，但作用不同：**

1. **`server/memory/`** = 训练流程编排层
   - 负责整个训练流程的编排
   - 数据准备和管理
   - 调用训练器

2. **`recall/`** = 训练器实现层
   - 包含具体的训练器实现
   - 执行实际的模型训练逻辑
   - 被 `server/memory/` 导入和使用

**路径配置**：
- `server/memory/training_service.py` 需要正确找到项目根目录下的 `recall/` 目录
- 当前路径计算：`os.path.dirname(os.path.dirname(os.path.dirname(__file__)))` 
  - `__file__` = `server/memory/training_service.py`
  - 往上3层 = 项目根目录
  - `recall_dir` = `项目根目录/recall`

