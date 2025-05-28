# 图像处理 Gradio UI

这个模块提供了基于 Presidio Image Redactor 的图像隐私信息检测与编辑的Web界面。

## 功能特性

### 🔧 主要功能

1. **OCR文字识别**: 使用 Tesseract OCR 从图像中提取文字内容
2. **PII检测**: 使用 Presidio 分析器检测提取文字中的个人隐私信息
3. **敏感信息编辑**: 在检测到PII的区域用指定颜色的矩形进行覆盖
4. **多语言支持**: 支持中文、英文等多种语言的识别和检测
5. **实时日志**: 显示处理过程中的详细日志信息

### 📝 支持的PII类型

- **PERSON**: 人名
- **PHONE_NUMBER**: 电话号码
- **EMAIL_ADDRESS**: 电子邮箱
- **ID_CARD**: 身份证号码
- **BANK_ACCOUNT**: 银行账号
- **CREDIT_CARD**: 信用卡号
- **LOCATION**: 地理位置
- **ORGANIZATION**: 组织机构
- **DATE_TIME**: 日期时间
- **IP_ADDRESS**: IP地址
- **URL**: 网址链接

## 使用方法

### 基本使用

1. **运行UI**:
   ```bash
   cd /Users/oren/MyFiles/Repository/all-pii-in-one
   python src/app/tabs/image_tab/test_ui.py
   ```

2. **访问界面**: 在浏览器中打开 `http://localhost:7863`

3. **上传图像**: 拖拽或点击上传包含文字的图像文件

4. **配置参数**:
   - 设置编辑颜色（RGB值）
   - 选择OCR识别语言
   - 选择PII检测语言
   - 指定输出文件名（可选）

5. **处理图像**: 点击"开始处理"按钮

6. **查看结果**: 在不同标签页中查看编辑后图像、统计信息和处理日志

### 编程接口

```python
from src.app.tabs.image_tab import ImageProcessorTab

# 创建处理器实例
processor = ImageProcessorTab()

# 创建Gradio界面
interface = processor.create_interface()

# 启动界面
interface.launch()
```

## 配置选项

### 编辑颜色设置

- **黑色**: R=0, G=0, B=0（默认，适合大多数情况）
- **白色**: R=255, G=255, B=255
- **红色**: R=255, G=0, B=0
- **自定义**: 可以设置任意RGB值

### 语言支持

**OCR语言**:
- `chi_sim`: 简体中文
- `eng`: 英语  
- `chi_tra`: 繁体中文

**PII检测语言**:
- `zh`: 中文
- `en`: 英文

## 输出结果

### 文件输出

- 编辑后的图像保存到 `data/image/output/` 目录
- 如果未指定文件名，将自动生成带时间戳的文件名
- 支持PNG、JPG等常见格式

### 界面显示

1. **编辑后图像**: 显示处理后的图像，敏感信息已被覆盖
2. **统计信息**: 显示图像尺寸、处理时间、配置参数等统计数据
3. **处理日志**: 显示详细的处理过程日志，帮助了解每个步骤的执行情况

## 性能说明

- **OCR识别**: 取决于图像大小和文字复杂度，通常需要2-10秒
- **PII检测**: 通常在1-2秒内完成
- **图像编辑**: 几乎实时完成
- **建议图像尺寸**: 不超过4096x4096像素以获得最佳性能

## 支持的图像格式

- PNG, JPG, JPEG, BMP, TIFF
- 分辨率: 建议不超过4096x4096
- 文件大小: 建议不超过10MB

## 示例图像

项目中提供了测试图像：
- `image.png` - 项目根目录下的示例图像
- `data/image/` 目录下的其他测试图像

## 故障排除

### 常见问题

1. **OCR识别不准确**:
   - 确保图像清晰度足够
   - 选择正确的OCR语言
   - 尝试调整图像对比度

2. **PII检测遗漏**:
   - 检查PII检测语言设置
   - 确认文字格式符合常见模式
   - 查看处理日志了解详细信息

3. **处理速度慢**:
   - 减小图像尺寸
   - 确保图像格式优化
   - 检查系统资源使用情况

### 日志查看

处理日志会实时显示在"处理日志"标签页中，包含：
- OCR识别进度
- PII检测结果
- 错误信息和警告
- 处理时间统计

## 依赖要求

确保安装了以下依赖：
- `presidio-image-redactor`
- `presidio-analyzer`
- `tesseract-ocr`
- `gradio`
- `PIL`

## API参考

查看 `image_tab.py` 中的 `ImageProcessorTab` 类了解完整的API接口。
