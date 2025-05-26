# Text Processor 模块

## 简介

Text Processor 是一个用于文本隐私信息处理的综合性模块，提供文本分割、PII（个人隐私信息）检测和匿名化等功能。该模块设计为模块化和可扩展的架构，可以灵活地添加新的识别器和匿名化策略。

## 目录结构

```
text_processor/
│
├── __init__.py                 # 模块初始化文件
├── processor.py                # 主处理器类，整合各个功能模块
│
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── analyzer.py             # PII 检测模块，基于 Presidio Analyzer
│   ├── anonymizer.py           # PII 匿名化模块，基于 Presidio Anonymizer
│   └── segmentation.py         # 文本分割模块
│
├── recognizers/                # 识别器模块
│   ├── __init__.py
│   ├── base.py                 # 基础识别器
│   │
│   ├── pattern/                # 基于模式的识别器
│   │   ├── __init__.py
│   │   ├── bank_card.py        # 银行卡号识别器
│   │   ├── car_plate.py        # 车牌号识别器
│   │   ├── id_card.py          # 身份证号识别器
│   │   ├── phone_number.py     # 电话号码识别器
│   │   └── url.py              # URL识别器
│   │
│   └── llm/                    # 基于大语言模型的识别器
│       ├── __init__.py
│       ├── recognizer.py       # LLM识别器主模块
│       ├── clients/            # LLM 客户端
│       │   ├── __init__.py
│       │   ├── base_client.py
│       │   ├── client_factory.py
│       │   └── gemini_client.py
│       ├── parsers/            # LLM 响应解析器
│       │   ├── __init__.py
│       │   ├── base_parser.py
│       │   ├── entity_match.py
│       │   ├── gemini_parser.py
│       │   └── parser_factory.py
│       └── prompts/            # LLM 提示词
│           ├── __init__.py
│           └── loader.py
│
└── utils/                      # 工具函数
    ├── __init__.py
    └── registry.py             # 识别器注册表
```

## 主要组件介绍

### 1. 主处理器 (processor.py)

`TextProcessor` 是整个模块的核心类，提供统一的接口来整合文本分析、匿名化和分割功能。

#### 核心功能

- **完整文本处理流程**: 提供一站式的PII检测和匿名化服务
- **灵活的配置选项**: 支持自定义处理策略和参数
- **批处理支持**: 支持批量处理多个文本
- **模块化设计**: 可以单独使用某个功能组件

#### 主要方法

```python
def process(
    self,
    text: str,
    enable_segmentation: bool = False,
    enable_analysis: bool = True,
    enable_anonymization: bool = True,
    language: Optional[str] = None,
    entities: Optional[List[str]] = None,
    score_threshold: Optional[float] = None,
    operators: Optional[Dict[str, str]] = None
) -> ProcessingResult
```

### 2. 核心功能模块 (core/)

包含所有核心功能的具体实现，分为三个主要子模块：

#### 检测模块 (analyzer.py)

`PresidioAnalyzer` 负责检测文本中的个人隐私信息 (PII)。

**特性：**
- 基于微软 Presidio 库
- 支持多语言处理
- 可扩展的识别器系统
- 灵活的置信度阈值设置

**支持的检测类型：**
- 身份证号码
- 电话号码
- 银行卡号
- 车牌号码
- URL链接
- 通过LLM增强的复杂PII检测

#### 匿名化模块 (anonymizer.py)

`PresidioAnonymizer` 负责将检测到的 PII 进行匿名化处理。

**支持的匿名化策略：**
- **mask**: 用指定字符掩码替换（如 `****`）
- **replace**: 用标签或假数据替换
- **redact**: 完全移除敏感信息
- **hash**: 对敏感信息进行哈希处理
- **encrypt**: 加密处理

#### 分割模块 (segmentation.py)

`TextSegmenter` 负责将文本分割为更小的处理单元。

**功能：**
- 句子级分割
- 段落级分割
- 自定义分割规则
- 保持原文结构信息

### 3. 识别器模块 (recognizers/)

包含所有用于识别特定类型 PII 的识别器。

#### 模式识别器 (pattern/)

基于正则表达式等模式匹配技术的识别器，专门针对中文环境进行优化。

**可用识别器：**

- **IDCardRecognizer**: 中国身份证号码识别
- **PhoneNumberRecognizer**: 中国手机号码识别
- **BankCardRecognizer**: 银行卡号识别
- **CarPlateNumberRecognizer**: 中国车牌号识别
- **URLRecognizer**: URL链接识别

#### LLM 识别器 (llm/)

基于大语言模型的识别器，可识别更复杂、非结构化的 PII。

**核心组件：**

- **recognizer.py**: LLM 识别器的核心实现
- **clients/**: LLM 服务客户端
  - 支持 Google Gemini
  - 可扩展支持其他LLM服务
- **parsers/**: LLM 响应解析器
  - 结构化解析LLM返回结果
  - 实体匹配和验证
- **prompts/**: LLM 提示词模板
  - 优化的中文PII检测提示词
  - 可配置的提示词模板

### 4. 工具函数 (utils/)

提供各种辅助功能和工具函数。

- **registry.py**: 识别器注册表管理

## 使用方法

### 基础使用

```python
from src.processors.text_processor import TextProcessor
from src.configs.processors.text_processor import TextProcessorConfig

# 使用默认配置
processor = TextProcessor()

# 处理文本
text = "我的身份证号是330102199901011234，电话是13800138000。"
result = processor.process(text)

# 输出匿名化后的文本
print(result.anonymized_text)
# 输出: "我的身份证号是****************，电话是***********。"
```

### 自定义配置

```python
from src.configs.processors.text_processor import TextProcessorConfig

# 创建自定义配置
config = TextProcessorConfig(
    analyzer=AnalyzerConfig(
        default_language="zh",
        default_score_threshold=0.6,
        enable_pattern_recognizers=True,
        enable_llm_recognizers=True
    ),
    anonymizer=AnonymizerConfig(
        default_operator="mask",
        mask_char="*"
    )
)

# 创建带有自定义配置的文本处理器
processor = TextProcessor(config=config)
```

### 高级用法

#### 分步处理

```python
# 仅进行PII检测
results = processor.analyze_only(text)
print(f"检测到 {len(results)} 个PII实体")

# 仅进行匿名化（需要预先分析的结果）
anonymized = processor.anonymize_only(text, results)

# 仅进行文本分割
segments = processor.segment_only(text)
```

#### 批量处理

```python
texts = [
    "张三的电话是13800138000",
    "李四的身份证是110101199001011234",
    "王五的邮箱是wang5@example.com"
]

results = processor.batch_process(texts)
for i, result in enumerate(results):
    print(f"文本 {i+1}: {result.anonymized_text}")
```

#### 灵活的处理选项

```python
# 自定义处理选项
result = processor.process(
    text=text,
    enable_segmentation=True,      # 启用文本分割
    enable_analysis=True,          # 启用PII分析
    enable_anonymization=True,     # 启用匿名化
    language="zh",                 # 指定语言
    entities=["PHONE_NUMBER", "ID_CARD"],  # 仅检测特定类型
    score_threshold=0.8,           # 提高置信度阈值
    operators={                    # 自定义匿名化策略
        "PHONE_NUMBER": "mask",
        "ID_CARD": "replace"
    }
)
```

### 处理结果结构

```python
@dataclass
class ProcessingResult:
    original_text: str                    # 原始文本
    segments: List[TextSegment]           # 分割后的文本段
    analysis_results: List[List[RecognizerResult]]  # 检测结果
    anonymized_segments: List[EngineResult]         # 匿名化结果
    anonymized_text: str                  # 最终匿名化文本
    metadata: Dict[str, Any]              # 元数据信息
```

## 扩展指南

### 1. 添加新的模式识别器

1. 在 `recognizers/pattern/` 目录下创建新的识别器文件：

```python
# recognizers/pattern/email.py
from presidio_analyzer import EntityRecognizer, Pattern

class EmailRecognizer(EntityRecognizer):
    def __init__(self):
        patterns = [
            Pattern(
                name="email_pattern",
                regex=r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                score=0.85
            )
        ]
        super().__init__(
            supported_entity="EMAIL",
            patterns=patterns,
            supported_language="zh"
        )
```

2. 在 `recognizers/pattern/__init__.py` 中导出新识别器
3. 在注册表中注册新识别器

### 2. 添加新的LLM客户端

1. 在 `recognizers/llm/clients/` 中创建新的客户端：

```python
# recognizers/llm/clients/openai_client.py
from .base_client import BaseLLMClient

class OpenAIClient(BaseLLMClient):
    def __init__(self, config):
        super().__init__(config)
        # 初始化OpenAI客户端
    
    async def generate_response(self, prompt: str) -> str:
        # 实现OpenAI API调用
        pass
```

2. 在客户端工厂中添加新客户端
3. 创建对应的解析器和配置

### 3. 自定义匿名化策略

可以通过配置不同的操作符来实现自定义匿名化策略：

```python
# 为不同实体类型配置不同策略
operators = {
    "PHONE_NUMBER": "mask",      # 电话号码用掩码
    "ID_CARD": "hash",          # 身份证号用哈希
    "EMAIL": "replace",         # 邮箱用标签替换
    "URL": "redact"             # URL完全移除
}

result = processor.process(text, operators=operators)
```

## 配置说明

### 分析器配置 (AnalyzerConfig)

```python
@dataclass
class AnalyzerConfig:
    presidio_enabled: bool = True
    default_language: str = "zh"
    supported_languages: List[str] = field(default_factory=lambda: ["zh", "en"])
    default_score_threshold: float = 0.6
    allowed_entities: Optional[List[str]] = None
    denied_entities: Optional[List[str]] = None
    enable_pattern_recognizers: bool = True
    enable_llm_recognizers: bool = False
```

### 匿名化器配置 (AnonymizerConfig)

```python
@dataclass
class AnonymizerConfig:
    presidio_enabled: bool = True
    default_operator: str = "mask"
    mask_char: str = "*"
    hash_type: str = "sha256"
    encrypt_key: Optional[str] = None
```

### 分割器配置 (SegmentationConfig)

```python
@dataclass
class SegmentationConfig:
    strategy: str = "sentence"  # "sentence", "paragraph", "custom"
    max_segment_length: int = 1000
    preserve_formatting: bool = True
```

## 依赖要求

### 核心依赖

```
presidio-analyzer>=2.2.0
presidio-anonymizer>=2.2.0
spacy>=3.4.0
```

### 语言模型

```bash
# 安装中文语言模型
python -m spacy download zh_core_web_sm

# 安装英文语言模型  
python -m spacy download en_core_web_sm
```

### 可选依赖

```
# LLM支持
google-generativeai>=0.3.0

# 其他工具
regex>=2022.0.0
```

## 性能考虑

### 1. 文本分割优化

- 对于长文本，启用分割可以提高处理效率
- 合理设置分割长度，避免过小的片段

### 2. 批处理优化

- 使用批处理模式处理大量文本
- 考虑使用异步处理提高吞吐量

### 3. LLM使用优化

- LLM识别器虽然准确率高，但处理速度较慢
- 建议先使用模式识别器，再用LLM处理复杂情况
- 合理设置LLM请求频率限制

### 4. 内存管理

- 处理大文本时注意内存使用
- 及时清理不需要的处理结果

## 故障排除

### 常见问题

1. **语言模型未安装**
   ```bash
   python -m spacy download zh_core_web_sm
   ```

2. **LLM API配置错误**
   - 检查API密钥配置
   - 确认网络连接正常

3. **识别器未正确注册**
   - 检查识别器是否正确继承基类
   - 确认注册表配置正确

4. **性能问题**
   - 考虑禁用LLM识别器
   - 调整置信度阈值
   - 使用文本分割

### 日志调试

模块提供详细的日志记录，可以通过以下方式启用调试日志：

```python
import logging
logging.getLogger('src.processors.text_processor').setLevel(logging.DEBUG)
```