# 文本处理UI模块

## 概述

本模块为 `test_text_processor.py` 提供了功能相同的 Gradio Web UI 界面，让用户可以通过友好的Web界面进行文本PII检测和匿名化处理。

## 功能特性

### 🎯 核心功能
- **PII检测**: 自动检测文本中的个人隐私信息
- **文本匿名化**: 对检测到的PII进行匿名化处理
- **实时处理**: 即时显示处理结果
- **多语言支持**: 支持中文、英文等多种语言

### 🛠 支持的PII类型
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
- **AGE**: 年龄
- **CURRENCY**: 货币金额

### ⚙️ 配置选项
- **文本分割**: 对长文本进行智能分割
- **语言选择**: 指定处理语言或自动检测
- **实体过滤**: 选择性检测特定类型的PII
- **置信度阈值**: 调整检测敏感度
- **匿名化策略**: 自定义匿名化方法

## 文件结构

```
text_tab/
├── __init__.py           # 模块初始化
├── text_tab.py          # 主UI模块
├── test_ui.py           # UI测试脚本
└── README.md            # 说明文档
```

## 使用方法

### 1. 独立运行文本UI

```bash
# 进入项目根目录
cd /Users/oren/MyFiles/Repository/all-pii-in-one

# 运行文本UI
PYTHONPATH=. python src/app/tabs/text_tab/text_tab.py
```

访问 http://localhost:7861 查看界面

### 2. 集成到主应用

```bash
# 运行主应用
PYTHONPATH=. python src/app/main_app.py
```

访问 http://localhost:7860 查看完整应用

### 3. 在代码中使用

```python
from src.app.tabs.text_tab import create_text_tab, TextProcessorTab

# 创建UI界面
interface = create_text_tab()

# 或者创建UI实例进行自定义
tab = TextProcessorTab()
interface = tab.create_interface()

# 启动界面
interface.launch()
```

## API说明

### TextProcessorTab 类

主要的UI处理类，封装了所有界面逻辑和文本处理功能。

#### 主要方法

- `__init__()`: 初始化UI实例
- `process_text()`: 处理文本并返回结果
- `get_example_text()`: 获取示例文本
- `create_interface()`: 创建Gradio界面

#### 处理参数

- `text`: 输入文本
- `enable_segmentation`: 是否启用文本分割
- `enable_analysis`: 是否启用PII分析
- `enable_anonymization`: 是否启用匿名化
- `language`: 处理语言 ("auto", "zh", "en")
- `entities_filter`: 实体类型过滤器（逗号分隔）
- `score_threshold`: 置信度阈值 (0.0-1.0)

#### 返回结果

- `original_text`: 原始文本
- `anonymized_text`: 匿名化后的文本
- `analysis_results`: 详细的PII检测结果
- `statistics`: 处理统计信息

## 与 test_text_processor.py 的对比

| 功能 | test_text_processor.py | text_tab UI |
|------|----------------------|-------------|
| PII检测 | ✅ 命令行输出 | ✅ Web界面展示 |
| 文本匿名化 | ✅ 控制台打印 | ✅ 实时显示 |
| 结果分析 | ✅ 日志输出 | ✅ 结构化展示 |
| 配置选项 | ❌ 硬编码 | ✅ 交互式配置 |
| 示例文本 | ✅ 预定义 | ✅ 一键加载 |
| 批量处理 | ❌ 单次处理 | ✅ 支持多次处理 |
| 用户友好性 | ❌ 需要编程知识 | ✅ 图形界面 |

## 技术实现

### 依赖项
- **Gradio**: Web UI框架
- **TextProcessor**: 核心文本处理引擎
- **AppConfig**: 配置管理系统

### 关键组件
1. **配置加载**: 从 `app_config.yaml` 加载处理器配置
2. **文本处理**: 使用 `TextProcessor` 执行核心处理逻辑
3. **结果格式化**: 将处理结果转换为用户友好的格式
4. **界面交互**: 处理用户输入和界面事件

### 错误处理
- 配置加载失败处理
- 文本处理异常捕获
- 用户输入验证
- 友好的错误信息显示

## 示例使用场景

### 1. 客服邮件处理
用户输入客服邮件，系统自动检测并匿名化其中的个人信息，保护客户隐私。

### 2. 文档脱敏
对包含敏感信息的文档进行批量脱敏处理，确保信息安全。

### 3. 数据分析准备
在进行数据分析前，先对原始数据进行匿名化处理，符合隐私保护要求。

### 4. 合规性检查
检查文档是否包含需要保护的个人信息，用于合规性审查。

## 故障排除

### 常见问题

1. **UI无法启动**
   - 检查是否安装了所有依赖项
   - 确认端口7861未被占用

2. **文本处理失败**
   - 检查配置文件是否正确
   - 确认TextProcessor组件初始化成功

3. **检测结果不准确**
   - 调整置信度阈值
   - 选择合适的语言设置
   - 使用实体类型过滤器

### 日志查看

```bash
# 查看应用日志
tail -f logs/src.app.tabs.text_tab.text_tab.log

# 查看文本处理器日志
tail -f logs/src.processors.text_processor.processor.log
```

## 后续开发

### 计划功能
- [ ] 文件上传支持
- [ ] 批量处理功能
- [ ] 结果导出功能
- [ ] 自定义匿名化规则
- [ ] 处理历史记录
- [ ] 性能监控界面

### 优化方向
- [ ] 响应速度优化
- [ ] 内存使用优化
- [ ] 并发处理支持
- [ ] 缓存机制优化

## 贡献指南

1. Fork 项目
2. 创建功能分支
3. 提交更改
4. 创建 Pull Request

## 许可证

请参考项目根目录的许可证文件。
