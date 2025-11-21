# 配置文件使用说明

## 概述

服务器现在支持通过配置文件来设置启动参数，方便随时修改配置而无需修改代码。

## 配置文件位置

默认配置文件：`server/config_qwen3vl.yaml`

## 配置文件格式

配置文件使用YAML格式，结构清晰易读：

```yaml
# 服务器配置
server:
  host: "0.0.0.0"  # 服务器监听地址
  port: 9999        # 服务器端口

# 模型配置
model:
  path: "./models/Qwen3-VL-8B-Instruct"  # 模型路径
  device: "cuda:0"  # 设备ID

# 生成参数配置
generation:
  max_new_tokens: 1000    # 最大生成token数
  temperature: 0.7       # 温度参数
  top_p: 0.9             # top-p采样参数
  top_k: 50              # top-k采样参数
  do_sample: true        # 是否使用采样

# 聊天历史配置
chat_history:
  max_history_length: 20  # 每个会话保留的最大历史消息数

# 日志配置
logging:
  level: "INFO"  # 日志级别
```

## 使用方法

### 1. 使用默认配置文件

直接启动服务器，会自动加载 `server/config_qwen3vl.yaml`：

```bash
cd /data0/user/ymdai/LLM_memory/qqbot_new/server
python api_server_qwen3vl.py
```

### 2. 指定自定义配置文件

如果配置文件在其他位置：

```bash
python api_server_qwen3vl.py --config /path/to/your/config.yaml
```

### 3. 命令行参数覆盖配置文件

命令行参数的优先级高于配置文件，可以临时覆盖配置：

```bash
# 使用配置文件，但临时修改端口
python api_server_qwen3vl.py --port 8888

# 使用配置文件，但临时修改设备
python api_server_qwen3vl.py --device cuda:1

# 同时修改多个参数
python api_server_qwen3vl.py --port 8888 --device cuda:1 --model-path ./models/Qwen3-VL-8B-Thinking
```

## 配置项说明

### 服务器配置 (server)

- `host`: 服务器监听地址
  - `"0.0.0.0"`: 监听所有网络接口（推荐）
  - `"127.0.0.1"`: 仅本地访问
  - 其他IP地址：监听指定接口

- `port`: 服务器端口号
  - 默认：`9999`
  - 确保端口未被占用

### 模型配置 (model)

- `path`: 模型路径
  - 可以是相对路径（相对于项目根目录）
  - 也可以是绝对路径
  - 示例：`"./models/Qwen3-VL-8B-Instruct"`

- `device`: 设备ID
  - `"cuda:0"`, `"cuda:1"` 等：使用指定GPU
  - `"cpu"`: 使用CPU（速度较慢）
  - `"auto"`: 自动选择（如果可用则使用GPU）

### 生成参数配置 (generation)

- `max_new_tokens`: 最大生成token数
  - 范围：建议 100-4096
  - 越大生成内容越长，但速度越慢

- `temperature`: 温度参数
  - 范围：0.0 - 2.0
  - 越低越确定，越高越随机
  - 推荐：0.7

- `top_p`: top-p采样参数（可选）
  - 范围：0.0 - 1.0
  - 控制采样的多样性
  - 推荐：0.9

- `top_k`: top-k采样参数（可选）
  - 范围：1 - 词汇表大小
  - 从概率最高的k个token中采样
  - 推荐：50

- `do_sample`: 是否使用采样
  - `true`: 使用采样（更随机）
  - `false`: 贪婪解码（更确定）
  - 当temperature > 0时建议设为true

### 聊天历史配置 (chat_history)

- `max_history_length`: 每个会话保留的最大历史消息数
  - 范围：建议 5-50
  - 越大上下文越多，但占用内存越多
  - 默认：20

### 日志配置 (logging)

- `level`: 日志级别
  - `DEBUG`: 最详细，包含所有调试信息
  - `INFO`: 一般信息（推荐）
  - `WARNING`: 仅警告和错误
  - `ERROR`: 仅错误
  - `CRITICAL`: 仅严重错误

## 修改配置示例

### 示例1：修改服务器端口

编辑 `config_qwen3vl.yaml`：

```yaml
server:
  host: "0.0.0.0"
  port: 8888  # 改为8888
```

### 示例2：切换到不同的模型

编辑 `config_qwen3vl.yaml`：

```yaml
model:
  path: "./models/Qwen3-VL-8B-Thinking"  # 切换到Thinking版本
  device: "cuda:0"
```

### 示例3：调整生成参数

编辑 `config_qwen3vl.yaml`：

```yaml
generation:
  max_new_tokens: 2000    # 生成更长的回复
  temperature: 0.8        # 更随机
  top_p: 0.95
  top_k: 50
  do_sample: true
```

### 示例4：增加聊天历史长度

编辑 `config_qwen3vl.yaml`：

```yaml
chat_history:
  max_history_length: 50  # 保留更多上下文
```

### 示例5：启用调试日志

编辑 `config_qwen3vl.yaml`：

```yaml
logging:
  level: "DEBUG"  # 查看详细日志
```

## 注意事项

1. **YAML格式**：注意缩进使用空格，不要使用Tab
2. **路径格式**：相对路径相对于项目根目录
3. **设备检查**：确保指定的GPU设备存在（使用`nvidia-smi`查看）
4. **端口占用**：确保端口未被其他程序占用
5. **配置文件不存在**：如果配置文件不存在，会使用默认配置并给出警告

## 依赖安装

如果提示缺少yaml模块，需要安装PyYAML：

```bash
pip install pyyaml
```

## 配置文件优先级

1. **命令行参数**（最高优先级）
2. **配置文件中的值**
3. **默认值**（如果配置文件不存在或缺少某项）

这样可以灵活地使用配置文件，同时保留命令行参数的灵活性。

