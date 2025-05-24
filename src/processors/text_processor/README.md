# Text Processor 模块

## 简介

Text Processor 是一个用于文本隐私信息处理的综合性模块，提供文本分割、PII（个人隐私信息）检测和匿名化等功能。该模块设计为模块化和可扩展的架构，可以灵活地添加新的识别器和匿名化策略。

## 目录结构

```
text_processor/
│
├── __init__.py                 # 模块初始化文件
├── api/                        # API 层，为外部提供接口
│   ├── __init__.py
│   └── processor.py            # 主处理器类，整合各个功能模块
│
├── core/                       # 核心功能模块
│   ├── __init__.py
│   ├── detection/              # PII 检测相关
│   │   ├── __init__.py
│   │   ├── detector.py         # 基础检测器
│   │   ├── presidio.py         # Presidio 检测器
│   │   └── pattern.py          # 模式检测器
│   │
│   ├── anonymization/          # 匿名化相关
│   │   ├── __init__.py
│   │   ├── engine.py           # 匿名化引擎
│   │   ├── operators/          # 匿名化操作符
│   │   │   ├── __init__.py
│   │   │   ├── base.py         # 基础操作符
│   │   │   ├── mask.py         # 掩码操作符
│   │   │   └── replace.py      # 替换操作符
│   │   └── strategies/         # 匿名化策略
│   │       ├── __init__.py
│   │       └── rule_based.py   # 基于规则的策略
│   │
│   └── segmentation/           # 文本分割相关
│       ├── __init__.py
│       └── segmenter.py        # 文本分割器
│
├── recognizers/                # 识别器模块
│   ├── __init__.py
│   ├── base.py                 # 基础识别器
│   ├── registry.py             # 识别器注册表
│   ├── pattern/                # 基于模式的识别器
│   │   ├── __init__.py
│   │   ├── bank_card.py
│   │   ├── car_plate.py
│   │   ├── id_card.py
│   │   ├── phone_number.py
│   │   └── url.py
│   │
│   └── llm/                    # 基于大语言模型的识别器
│       ├── __init__.py
│       ├── recognizer.py
│       ├── clients/            # LLM 客户端
│       ├── parsers/            # LLM 响应解析器
│       └── prompts/            # LLM 提示词
│
└── utils/                      # 工具函数
    ├── __init__.py
    ├── validation.py           # 验证功能
    └── constants.py            # 常量定义
```

## 主要组件介绍

### 1. API 层 (api/)

API 层为外部提供统一的接口，是用户与 Text Processor 模块交互的主要入口点。

- `processor.py`: 主处理器类，整合检测、匿名化和分割等功能，提供简洁的 API 调用方式。

### 2. 核心功能模块 (core/)

包含所有核心功能的具体实现，分为三个主要子模块：

#### 检测模块 (detection/)

负责检测文本中的个人隐私信息 (PII)。

- `detector.py`: 定义基础检测器接口和通用功能
- `presidio.py`: 基于微软 Presidio 库的检测实现
- `pattern.py`: 基于自定义模式匹配的检测实现

#### 匿名化模块 (anonymization/)

负责将检测到的 PII 进行匿名化处理。

- `engine.py`: 匿名化引擎，协调不同的匿名化策略和操作符
- `operators/`: 包含各种匿名化操作符的实现
  - `base.py`: 操作符基类
  - `mask.py`: 掩码操作符（如用 * 替换敏感信息）
  - `replace.py`: 替换操作符（如用标签或假数据替换敏感信息）
- `strategies/`: 包含各种匿名化策略的实现
  - `rule_based.py`: 基于规则的匿名化策略

#### 分割模块 (segmentation/)

负责将文本分割为更小的处理单元，如句子或段落。

- `segmenter.py`: 文本分割器的实现

### 3. 识别器模块 (recognizers/)

包含所有用于识别特定类型 PII 的识别器。

- `base.py`: 识别器基类，定义通用接口和功能
- `registry.py`: 识别器注册表，管理所有可用的识别器

#### 模式识别器 (pattern/)

基于正则表达式等模式匹配技术的识别器。

- `bank_card.py`: 银行卡号识别器
- `car_plate.py`: 车牌号识别器
- `id_card.py`: 身份证号识别器
- `phone_number.py`: 电话号码识别器
- `url.py`: URL 识别器

#### LLM 识别器 (llm/)

基于大语言模型的识别器，可识别更复杂、非结构化的 PII。

- `recognizer.py`: LLM 识别器的核心实现
- `clients/`: LLM 服务客户端
- `parsers/`: LLM 响应解析器
- `prompts/`: LLM 提示词模板

### 4. 工具函数 (utils/)

提供各种辅助功能和工具函数。

- `validation.py`: 数据验证相关功能
- `constants.py`: 常量定义

### 5. 配置模块 (config/)

管理模块的配置信息。

- `settings.py`: 配置设置和默认值

## 使用方法

```python
from src.processors.text_processor.api.processor import TextProcessor
from src.processors.text_processor.config.settings import ProcessorConfig

# 创建自定义配置
config = ProcessorConfig(
    anonymize_strategy="mask",    # 使用掩码策略
    mask_char="*",                # 掩码字符
    language="zh",                # 语言设置
    use_llm=True                  # 使用LLM增强识别
)

# 创建带有自定义配置的文本处理器
processor = TextProcessor(config=config)

# 处理文本
text = "我的身份证号是330102199901011234，电话是13800138000。"
result = processor.process(text)

# 输出匿名化后的文本
print(result.anonymized_text)
# 输出: "我的身份证号是****************，电话是************。"
```

## 扩展指南

### 1. 添加新的模式识别器

1. 在 `recognizers/pattern/` 目录下创建新的识别器文件（例如 `email.py`）
2. 实现识别器类，继承自 `EntityRecognizer`：

```python
from presidio_analyzer import EntityRecognizer, Pattern
from presidio_analyzer.nlp_engine import NlpArtifacts

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
    
    def analyze(self, text, entities, nlp_artifacts=None):
        results = super().analyze(text, entities, nlp_artifacts)
        return results
```

3. 在 `recognizers/pattern/__init__.py` 中导出新的识别器
4. 在 `recognizers/registry.py` 中注册新的识别器

### 2. 添加新的匿名化操作符

1. 在 `core/anonymization/operators/` 目录下创建新的操作符文件
2. 实现操作符类，继承自基础操作符
3. 在 `core/anonymization/operators/__init__.py` 中导出新的操作符
4. 在 `core/anonymization/engine.py` 中注册新的操作符

### 3. 添加新的 LLM 识别器

1. 在 `recognizers/llm/clients/` 中添加新的 LLM 客户端（如需）
2. 在 `recognizers/llm/prompts/` 中添加相应的提示词模板
3. 在 `recognizers/llm/recognizer.py` 中实现新的识别逻辑


## 注意事项

- **性能优化**：处理大量文本时，建议使用批处理模式以提高效率
- **语言支持**：目前主要支持中文，其他语言的支持需要添加相应的识别器
- **LLM配置**：使用LLM识别器需要正确配置API密钥和服务地址
- **扩展性**：添加新的识别器或匿名化策略时，需要同时更新相应的测试
- **依赖库**：确保所有依赖库（如presidio-analyzer和presidio-anonymizer）已正确安装