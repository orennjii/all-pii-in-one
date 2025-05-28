# API 参考文档

本文档详细描述了 ALL PII IN ONE 系统的所有 API 接口，包括使用方法、参数说明和返回值格式。

## 目录

- [概述](#概述)
- [文本处理API](#文本处理api)
- [音频处理API](#音频处理api)
- [配置管理API](#配置管理api)
- [错误处理](#错误处理)
- [使用示例](#使用示例)

## 概述

ALL PII IN ONE 提供了两种主要的API接口：

1. **Python API**: 用于直接在Python代码中集成PII处理功能
2. **REST API**: 用于通过HTTP请求调用服务（基于Gradio）

## 文本处理API

### TextProcessor 类

文本PII检测和匿名化的主要接口。

#### 初始化

```python
from src.processors.text_processor import TextProcessor
from src.configs import AppConfig

# 从配置文件初始化
config = AppConfig.from_yaml_file("config/app_config.yaml")
processor = TextProcessor(config.processor.text_processor)

# 或者直接传入配置字典
processor = TextProcessor({
    "supported_entities": ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS"],
    "analyzer": {
        "presidio_enabled": True,
        "supported_languages": ["zh", "en"],
        "enable_llm_recognizers": True
    }
})
```

#### 核心方法

##### process()

处理文本中的PII实体。

```python
def process(
    self, 
    text: str,
    entities: Optional[List[str]] = None,
    anonymization_strategy: Optional[str] = None,
    language: Optional[str] = None
) -> ProcessingResult
```

**参数:**
- `text` (str): 要处理的文本内容
- `entities` (List[str], optional): 要检测的实体类型列表。如果为None，使用配置中的默认实体
- `anonymization_strategy` (str, optional): 匿名化策略 ("replace", "mask", "hash", "redact")
- `language` (str, optional): 文本语言代码 ("zh", "en")

**返回值:**
- `ProcessingResult`: 包含处理结果的对象

**使用示例:**

```python
# 基本使用
text = "我是张三，电话号码是13812345678，邮箱是zhangsan@example.com"
result = processor.process(text)

print(f"原文本: {result.original_text}")
print(f"匿名化文本: {result.anonymized_text}")
print(f"检测到的实体: {len(result.analysis_results[0])}")

# 指定特定实体类型
result = processor.process(
    text,
    entities=["PERSON", "PHONE_NUMBER"],
    anonymization_strategy="mask"
)

# 指定语言
result = processor.process(
    "My name is John Doe and my phone is +1-555-123-4567",
    language="en"
)
```

##### analyze()

仅进行PII实体分析，不执行匿名化。

```python
def analyze(
    self,
    text: str,
    entities: Optional[List[str]] = None,
    language: Optional[str] = None
) -> List[RecognizerResult]
```

**参数:**
- `text` (str): 要分析的文本
- `entities` (List[str], optional): 要检测的实体类型
- `language` (str, optional): 文本语言

**返回值:**
- `List[RecognizerResult]`: 检测到的PII实体列表

**使用示例:**

```python
text = "联系人：张三，手机：13812345678"
entities = processor.analyze(text)

for entity in entities:
    print(f"实体类型: {entity.entity_type}")
    print(f"文本: '{text[entity.start:entity.end]}'")
    print(f"位置: {entity.start}-{entity.end}")
    print(f"置信度: {entity.score}")
    print("---")
```

##### anonymize()

对文本进行匿名化处理。

```python
def anonymize(
    self,
    text: str,
    analyzer_results: List[RecognizerResult],
    anonymization_strategy: Optional[str] = None
) -> EngineResult
```

### ProcessingResult 类

文本处理的结果对象。

#### 属性

```python
@dataclass
class ProcessingResult:
    original_text: str                           # 原始文本
    segments: List[TextSegment]                  # 文本分段
    analysis_results: List[List[RecognizerResult]] # 分析结果
    anonymized_segments: List[EngineResult]      # 匿名化分段
    anonymized_text: str                         # 最终匿名化文本
    metadata: Optional[Dict[str, Any]] = None    # 元数据
```

#### 方法

##### to_dict()

将结果转换为字典格式。

```python
def to_dict(self) -> Dict[str, Any]
```

##### get_entities_summary()

获取检测到的实体摘要。

```python
def get_entities_summary(self) -> Dict[str, int]
```

**返回示例:**
```python
{
    "PERSON": 2,
    "PHONE_NUMBER": 1,
    "EMAIL_ADDRESS": 1,
    "total": 4
}
```

## 音频处理API

### AudioProcessor 类

音频PII检测和匿名化的主要接口。

#### 初始化

```python
from src.processors.audio_processor import AudioProcessor
from src.configs import AppConfig

config = AppConfig.from_yaml_file("config/app_config.yaml")
processor = AudioProcessor(config)
```

#### 核心方法

##### process_audio()

处理音频文件中的PII内容。

```python
def process_audio(
    self,
    audio_path: str,
    enable_diarization: bool = True,
    enable_pii_detection: bool = True,
    anonymization_method: str = "beep",
    output_path: Optional[str] = None
) -> AudioProcessingResult
```

**参数:**
- `audio_path` (str): 输入音频文件路径
- `enable_diarization` (bool): 是否启用说话人分离
- `enable_pii_detection` (bool): 是否启用PII检测
- `anonymization_method` (str): 匿名化方法 ("beep", "silence", "voice_conversion")
- `output_path` (str, optional): 输出文件路径

**返回值:**
- `AudioProcessingResult`: 音频处理结果

**使用示例:**

```python
# 基本音频处理
result = processor.process_audio("input.wav")

print(f"转录文本: {result.transcription_result.segments[0].text}")
print(f"说话人数量: {result.diarization_result.num_speakers}")
print(f"检测到的PII: {len(result.pii_detection_result.entities)}")
print(f"输出文件: {result.anonymized_audio_path}")

# 指定匿名化方法
result = processor.process_audio(
    "input.wav",
    anonymization_method="voice_conversion",
    output_path="output_anonymous.wav"
)

# 仅转录，不进行匿名化
result = processor.process_audio(
    "input.wav",
    enable_pii_detection=False
)
```

##### transcribe()

仅进行语音转录。

```python
def transcribe(
    self,
    audio_path: str,
    language: Optional[str] = None
) -> TranscriptionResult
```

##### diarize()

仅进行说话人分离。

```python
def diarize(
    self,
    audio_path: str
) -> DiarizationResult
```

### AudioProcessingResult 类

音频处理的结果对象。

```python
@dataclass
class AudioProcessingResult:
    original_audio_path: str                    # 原始音频路径
    transcription_result: TranscriptionResult  # 转录结果
    diarization_result: Optional[DiarizationResult] # 说话人分离结果
    pii_detection_result: Optional[AudioPIIResult]  # PII检测结果
    anonymized_audio_path: Optional[str]        # 匿名化音频路径
```

## 配置管理API

### AppConfig 类

应用程序配置管理。

#### 类方法

##### from_yaml_file()

从YAML文件加载配置。

```python
@classmethod
def from_yaml_file(cls, file_path: str) -> 'AppConfig'
```

##### from_dict()

从字典创建配置。

```python
@classmethod
def from_dict(cls, config_dict: Dict[str, Any]) -> 'AppConfig'
```

#### 实例方法

##### to_dict()

转换为字典格式。

```python
def to_dict(self) -> Dict[str, Any]
```

##### validate()

验证配置的有效性。

```python
def validate(self) -> bool
```

##### save_to_file()

保存配置到文件。

```python
def save_to_file(self, file_path: str) -> None
```

**使用示例:**

```python
# 加载配置
config = AppConfig.from_yaml_file("config/app_config.yaml")

# 修改配置
config.general.log_level = "DEBUG"
config.processor.text_processor.analyzer.default_score_threshold = 0.7

# 验证配置
if config.validate():
    print("配置有效")

# 保存配置
config.save_to_file("config/modified_config.yaml")
```

## 错误处理

### 异常类型

系统定义了以下异常类型：

#### PIIProcessingError

PII处理相关的基础异常。

```python
class PIIProcessingError(Exception):
    """PII处理错误基类"""
    pass
```

#### ConfigurationError

配置相关异常。

```python
class ConfigurationError(PIIProcessingError):
    """配置错误"""
    pass
```

#### ModelLoadError

模型加载异常。

```python
class ModelLoadError(PIIProcessingError):
    """模型加载错误"""
    pass
```

#### AudioProcessingError

音频处理异常。

```python
class AudioProcessingError(PIIProcessingError):
    """音频处理错误"""
    pass
```

### 错误处理示例

```python
from src.processors.exceptions import PIIProcessingError, AudioProcessingError

try:
    result = processor.process_audio("invalid_file.wav")
except AudioProcessingError as e:
    print(f"音频处理错误: {e}")
except PIIProcessingError as e:
    print(f"PII处理错误: {e}")
except Exception as e:
    print(f"未知错误: {e}")
```

## 使用示例

### 批量处理文本

```python
from src.processors.text_processor import TextProcessor
from src.configs import AppConfig
import json

def batch_process_texts(texts: List[str], output_file: str):
    """批量处理文本列表"""
    config = AppConfig.from_yaml_file("config/app_config.yaml")
    processor = TextProcessor(config.processor.text_processor)
    
    results = []
    for i, text in enumerate(texts):
        try:
            result = processor.process(text)
            results.append({
                "id": i,
                "original": result.original_text,
                "anonymized": result.anonymized_text,
                "entities_count": len(result.analysis_results[0])
            })
        except Exception as e:
            results.append({
                "id": i,
                "error": str(e)
            })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

# 使用示例
texts = [
    "张三的电话是13812345678",
    "联系邮箱：user@example.com",
    "身份证号：110101199001011234"
]
batch_process_texts(texts, "batch_results.json")
```

### 音频处理流水线

```python
from src.processors.audio_processor import AudioProcessor
from src.configs import AppConfig
from pathlib import Path

def process_audio_directory(input_dir: str, output_dir: str):
    """处理目录中的所有音频文件"""
    config = AppConfig.from_yaml_file("config/app_config.yaml")
    processor = AudioProcessor(config)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for audio_file in input_path.glob("*.wav"):
        try:
            print(f"处理文件: {audio_file.name}")
            
            result = processor.process_audio(
                str(audio_file),
                anonymization_method="beep",
                output_path=str(output_path / f"anon_{audio_file.name}")
            )
            
            # 保存转录结果
            transcript_file = output_path / f"{audio_file.stem}_transcript.json"
            with open(transcript_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "segments": [seg.to_dict() for seg in result.transcription_result.segments],
                    "pii_entities": [entity.to_dict() for entity in result.pii_detection_result.entities]
                }, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"处理文件 {audio_file.name} 时出错: {e}")

# 使用示例
process_audio_directory("data/audio/input", "data/audio/output")
```

### 自定义PII实体识别

```python
from presidio_analyzer import Pattern, PatternRecognizer
from src.processors.text_processor import TextProcessor

class ChineseIdCardRecognizer(PatternRecognizer):
    """中国身份证识别器"""
    
    PATTERNS = [
        Pattern("CHINESE_ID_CARD", r"\d{17}[\dXx]", 0.8)
    ]
    
    def __init__(self):
        super().__init__(
            supported_entity="CHINESE_ID_CARD",
            patterns=self.PATTERNS,
            context=["身份证", "证件号", "ID"]
        )

# 注册自定义识别器
def create_custom_processor():
    config = AppConfig.from_yaml_file("config/app_config.yaml")
    processor = TextProcessor(config.processor.text_processor)
    
    # 添加自定义识别器
    processor.analyzer.add_recognizer(ChineseIdCardRecognizer())
    
    return processor

# 使用自定义处理器
processor = create_custom_processor()
text = "我的身份证号是110101199001011234"
result = processor.process(text)
print(result.anonymized_text)
```

### 集成到Web应用

```python
from flask import Flask, request, jsonify
from src.processors.text_processor import TextProcessor
from src.configs import AppConfig

app = Flask(__name__)

# 初始化处理器
config = AppConfig.from_yaml_file("config/app_config.yaml")
text_processor = TextProcessor(config.processor.text_processor)

@app.route('/api/process_text', methods=['POST'])
def process_text_api():
    """文本处理API端点"""
    try:
        data = request.json
        text = data.get('text', '')
        entities = data.get('entities', None)
        strategy = data.get('anonymization_strategy', 'replace')
        
        if not text:
            return jsonify({'error': 'Text is required'}), 400
        
        result = text_processor.process(
            text,
            entities=entities,
            anonymization_strategy=strategy
        )
        
        return jsonify({
            'original_text': result.original_text,
            'anonymized_text': result.anonymized_text,
            'entities_summary': result.get_entities_summary(),
            'success': True
        })
        
    except Exception as e:
        return jsonify({'error': str(e), 'success': False}), 500

if __name__ == '__main__':
    app.run(debug=True)
```

这个API文档提供了完整的接口说明和使用示例，帮助开发者快速集成和使用PII处理功能。
