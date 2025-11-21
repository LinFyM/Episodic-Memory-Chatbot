# 多GPU支持使用指南

## 概述

本系统已全面升级为支持多GPU并行处理，以降低单卡显存压力。通过Hugging Face Accelerate库实现自动设备分配和数据并行。

## 配置方式

### 1. 自动多GPU（推荐）

在 `server/config_qwen3vl.yaml` 中设置：

```yaml
model:
  device: "auto"  # 自动检测并使用所有可用GPU
  multi_gpu:
    enabled: true
    max_memory: null  # 可选：限制每张GPU内存使用，如 {"cuda:0": "10GB"}
    gradient_accumulation_steps: 1  # 梯度累积步数，增加可降低显存使用
```

### 2. 手动指定GPU列表

```yaml
model:
  device: ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]  # 指定使用哪些GPU
  multi_gpu:
    enabled: true
    gradient_accumulation_steps: 2  # 2张GPU时可设置为2，降低显存
```

### 3. 单GPU模式（默认）

```yaml
model:
  device: "cuda:0"
  multi_gpu:
    enabled: false
```

## 功能特性

### 推理阶段
- **自动设备分配**：使用 `device_map="auto"` 让Accelerate自动分配模型层到不同GPU
- **负载均衡**：根据GPU内存自动平衡负载
- **无缝切换**：无需修改API调用代码

### 训练阶段
- **数据并行**：支持DDP（Distributed Data Parallel）
- **梯度累积**：可配置梯度累积步数以降低显存使用
- **混合精度**：使用bf16混合精度训练
- **单模型实例**：训练过程中只加载一个模型实例

## 使用方法

### 1. 启动服务器

```bash
# 多GPU模式
python server/api_server_qwen3vl.py --config server/config_qwen3vl.yaml

# 或直接运行（使用配置文件中的设置）
python server/api_server_qwen3vl.py
```

### 2. 训练时的显存优化

- **梯度累积**：增加 `gradient_accumulation_steps` 可以显著降低显存使用
- **批次大小**：适当减小批次大小
- **LoRA参数**：使用较小的 `lora_r` 值

### 3. 推荐配置

对于4张GPU的系统：

```yaml
model:
  device: "auto"
  multi_gpu:
    enabled: true
    gradient_accumulation_steps: 4  # 与GPU数量匹配

training:
  training_config:
    batch_size: 1  # 减小批次大小
    lora_config:
      r: 4  # 减小LoRA rank
```

## 技术实现

### 模型加载
- 使用 `device_map="auto"` 实现自动设备分配
- 支持 `max_memory` 参数限制每张GPU内存使用
- 兼容单GPU和多GPU环境

### 数据并行
- 基于Hugging Face Accelerate实现
- 支持梯度累积以降低显存压力
- 自动处理设备间的数据传输

### 内存管理
- 训练前强制清理显存
- 使用 `torch.cuda.empty_cache()` 和 `torch.cuda.synchronize()`
- 多重垃圾回收确保内存释放

## 故障排除

### 1. CUDA内存不足
```yaml
# 增加梯度累积步数
multi_gpu:
  gradient_accumulation_steps: 8

# 减小批次大小
training:
  training_config:
    batch_size: 1
```

### 2. 设备分配不均
```yaml
# 手动指定内存限制
multi_gpu:
  max_memory:
    cuda:0: "8GB"
    cuda:1: "8GB"
```

### 3. DDP启动问题
确保使用正确的启动方式：
```bash
# 单进程模式（推荐）
python server/api_server_qwen3vl.py

# 如果需要DDP，使用torchrun
torchrun --nproc_per_node=4 server/api_server_qwen3vl.py
```

## 性能优化建议

1. **显存优化**：
   - 优先增加 `gradient_accumulation_steps`
   - 其次减小 `batch_size`
   - 最后减小 `lora_r`

2. **训练速度**：
   - 使用更多GPU可以显著加速训练
   - 梯度累积会略微降低速度但节省显存

3. **推理性能**：
   - 多GPU可以提高并发处理能力
   - 自动负载均衡确保高效利用

## 注意事项

- 多GPU模式需要GPU间高速互联（NVLink）
- 确保所有GPU驱动版本一致
- 监控各GPU温度和使用率
- 定期清理显存避免碎片化
