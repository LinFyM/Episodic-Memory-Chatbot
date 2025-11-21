# 文件处理功能安装指南

## 1. Python 依赖包安装

### 基础安装（推荐）
```bash
pip install pdfminer.six>=20221105 pdf2image>=1.16.0 python-docx>=0.8.11 beautifulsoup4>=4.11.0
```

### 完整安装（包含OCR支持）
```bash
pip install pdfminer.six>=20221105 pdf2image>=1.16.0 python-docx>=0.8.11 pytesseract>=0.3.10 beautifulsoup4>=4.11.0
```

### 或使用 requirements.txt
```bash
pip install -r server/requirements.txt
```

## 2. 系统依赖安装

### PDF 图片提取需要 poppler-utils

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install poppler-utils
```

**CentOS/RHEL:**
```bash
sudo yum install poppler-utils
# 或对于较新版本
sudo dnf install poppler-utils
```

**macOS:**
```bash
brew install poppler
```

**Windows:**
1. 下载 poppler for Windows: https://github.com/oschwartz10612/poppler-windows/releases/
2. 解压并添加到系统 PATH 环境变量

### OCR 功能需要 tesseract（可选）

**Ubuntu/Debian:**
```bash
sudo apt-get install tesseract-ocr tesseract-ocr-chi-sim  # 包含中文支持
```

**CentOS/RHEL:**
```bash
sudo yum install tesseract tesseract-langpack-chi_sim
```

**macOS:**
```bash
brew install tesseract tesseract-lang
```

**Windows:**
1. 下载安装程序: https://github.com/UB-Mannheim/tesseract/wiki
2. 安装后添加到系统 PATH

## 3. 验证安装

### 验证 Python 包
```python
python -c "import pdfminer; import pdf2image; import docx; print('✅ 所有Python包已安装')"
```

### 验证系统工具
```bash
# 验证 poppler
pdftotext -v  # 或 pdfinfo -v

# 验证 tesseract（如果安装了）
tesseract --version
```

## 4. 功能说明

### 支持的文件类型

1. **PDF文件** (.pdf)
   - 文本提取：使用 `pdfminer.six` 或 `pdftotext`
   - 图片提取：使用 `pdf2image`（需要 poppler-utils）

2. **Word文档** (.docx)
   - 文本提取：使用 `python-docx`
   - 图片提取：从文档中提取嵌入的图片

3. **文本文件** (.txt, .md, .log)
   - 直接读取文本内容

4. **其他文件**
   - 尝试使用 OCR（需要 pytesseract 和 tesseract）

### 处理流程

1. 接收文件：通过 `[CQ:file]` 接收文件URL
2. 下载文件：保存到服务器的 `uploaded_files/` 目录
3. 提取内容：
   - 文本内容提取并添加到消息文本中
   - 图片内容提取并保存到 `uploaded_images/` 目录
4. 构建多模态消息：
   - 文本部分：原始消息 + 文件文本内容
   - 图片部分：原始图片 + 从文件中提取的图片
5. 输入模型：作为多模态消息输入给 Qwen3-VL 模型

## 5. 注意事项

1. **文件大小限制**：大文件下载和处理可能耗时，建议设置合理的超时时间
2. **内存使用**：PDF转图片可能占用较多内存，特别是页数较多的PDF
3. **中断支持**：文件处理过程中支持消息打断机制，新消息到达时会中断当前文件处理
4. **错误处理**：如果某个文件处理失败，会记录警告日志并继续处理其他文件

