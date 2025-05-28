# ALL PII IN ONE - 综合个人身份信息处理系统

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Gradio](https://img.shields.io/badge/gradio-web%20ui-orange.svg)](https://gradio.app/)

一个强大的全方位个人身份信息（PII）检测、分析和匿名化处理系统。支持文本、音频、图像和视频多种媒体类型的PII处理，提供直观的Web界面和灵活的配置选项。

## 🌟 主要特性

### 🎯 多模态PII处理
- **文本处理**: 基于 Presidio 的高精度 PII 检测和匿名化
- **音频处理**: 语音转录、说话人分离、语音匿名化和转换
- **图像处理**: 面部识别和图像匿名化（开发中）
- **视频处理**: 综合视频内容的PII处理（开发中）

### 🤖 AI驱动的智能识别
- **LLM集成**: 支持 Gemini 等大语言模型进行智能PII识别
- **多语言支持**: 中文、英文等多语言PII检测
- **自定义规则**: 灵活的模式识别和自定义实体配置

### 🎙️ 专业音频处理
- **高质量转录**: 基于 WhisperX 的准确语音识别
- **说话人分离**: 自动识别和分离不同说话人
- **语音转换**: 基于 Seed-VC 的高质量语音转换
- **音频匿名化**: 多种匿名化策略（蜂鸣音、静音、语音转换）

### 🔧 高度可配置
- **分层配置系统**: YAML 配置文件支持灵活参数调整
- **模块化架构**: 独立的处理器模块便于扩展和维护
- **性能优化**: 支持 GPU 加速和批处理

## 📋 目录

- [系统架构](#系统架构)
- [安装说明](#安装说明)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [使用指南](#使用指南)
- [API文档](#api文档)
- [开发指南](#开发指南)
- [常见问题](#常见问题)
- [贡献指南](#贡献指南)

## 🏗️ 系统架构

### 整体架构图

```mermaid
graph TB
    subgraph "用户界面层 (UI Layer)"
        A[Gradio Web UI]
        A1[文本处理界面]
        A2[音频处理界面]
        A3[图像处理界面]
        A4[视频处理界面]
        A --> A1
        A --> A2
        A --> A3
        A --> A4
    end

    subgraph "应用层 (Application Layer)"
        B[主应用 main.py]
        B1[路由管理]
        B2[会话管理]
        B3[文件管理]
        B --> B1
        B --> B2
        B --> B3
    end

    subgraph "处理器层 (Processor Layer)"
        C1[文本处理器<br/>TextProcessor]
        C2[音频处理器<br/>AudioProcessor]
        C3[图像处理器<br/>ImageProcessor]
        C4[视频处理器<br/>VideoProcessor]
    end

    subgraph "核心组件层 (Core Components)"
        D1[PII分析器<br/>PresidioAnalyzer]
        D2[PII匿名化器<br/>PresidioAnonymizer]
        D3[文本分割器<br/>TextSegmenter]
        D4[语音转录器<br/>WhisperXTranscriber]
        D5[说话人分离器<br/>PyannoteAudioDiarizer]
        D6[语音转换器<br/>VoiceConverter]
        D7[音频匿名化器<br/>VoiceAnonymizer]
    end

    subgraph "AI模型层 (AI Models Layer)"
        E1[LLM模型<br/>Gemini/ChatGPT]
        E2[Whisper模型<br/>语音识别]
        E3[Pyannote模型<br/>说话人分离]
        E4[Seed-VC模型<br/>语音转换]
        E5[Presidio模型<br/>NLP识别]
    end

    subgraph "配置层 (Configuration Layer)"
        F1[应用配置<br/>app_config.yaml]
        F2[模型配置]
        F3[处理器配置]
        F4[环境变量配置]
    end

    subgraph "存储层 (Storage Layer)"
        G1[临时文件存储]
        G2[模型缓存]
        G3[结果缓存]
        G4[参考音频库]
    end

    A1 --> C1
    A2 --> C2
    A3 --> C3
    A4 --> C4

    C1 --> D1
    C1 --> D2
    C1 --> D3
    C2 --> D4
    C2 --> D5
    C2 --> D6
    C2 --> D7

    D1 --> E1
    D1 --> E5
    D4 --> E2
    D5 --> E3
    D6 --> E4

    C1 -.-> F1
    C2 -.-> F1
    C3 -.-> F1
    C4 -.-> F1

    D4 -.-> G1
    D5 -.-> G2
    D6 -.-> G3
    D7 -.-> G4
```

### 音频处理序列图

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as Gradio UI
    participant AP as AudioProcessor
    participant TR as WhisperX转录器
    participant DI as Pyannote分离器
    participant PII as PII检测器
    participant AN as 音频匿名化器
    participant VC as 语音转换器

    User->>UI: 上传音频文件
    UI->>AP: 处理音频请求
    
    AP->>AP: 验证音频格式
    AP->>TR: 启动语音转录
    TR->>TR: 音频预处理
    TR->>TR: Whisper模型推理
    TR-->>AP: 返回转录结果
    
    par 并行处理
        AP->>DI: 说话人分离
        DI->>DI: Pyannote模型推理
        DI-->>AP: 返回说话人时间段
    and
        AP->>PII: PII检测
        PII->>PII: 文本PII分析
        PII-->>AP: 返回PII实体
    end
    
    AP->>AP: 整合结果，确定需匿名化片段
    
    alt 选择匿名化策略
        AP->>AN: 蜂鸣音替换
        AN-->>AP: 返回匿名化音频
    else
        AP->>VC: 语音转换
        VC->>VC: Seed-VC模型推理
        VC-->>AP: 返回转换后音频
    end
    
    AP-->>UI: 返回处理结果
    UI-->>User: 展示结果和播放匿名化音频
```

### 音频处理流程图

```mermaid
flowchart TD
    A[输入音频文件] --> B[音频格式验证]
    B --> C{格式支持?}
    C -->|否| C1[格式转换]
    C1 --> D[音频预处理]
    C -->|是| D[音频预处理]
    
    D --> E[音频质量检测]
    E --> F{质量符合要求?}
    F -->|否| F1[音频增强/降噪]
    F1 --> G[语音转录]
    F -->|是| G[语音转录]
    
    G --> H[WhisperX模型推理]
    H --> I[生成转录文本]
    I --> J[文本对齐处理]
    
    J --> K{启用说话人分离?}
    K -->|是| L[Pyannote模型推理]
    K -->|否| M[PII实体检测]
    
    L --> L1[生成说话人时间段]
    L1 --> L2[说话人标注]
    L2 --> M[PII实体检测]
    
    M --> N[Presidio文本分析]
    N --> O{启用LLM检测?}
    O -->|是| P[LLM智能识别]
    O -->|否| Q[实体标注和定位]
    
    P --> P1[Gemini模型推理]
    P1 --> P2[结果融合]
    P2 --> Q[实体标注和定位]
    
    Q --> R[时间轴对齐]
    R --> S[生成匿名化方案]
    
    S --> T{选择匿名化策略}
    
    T -->|蜂鸣音| U[蜂鸣音生成]
    U --> U1[音频片段替换]
    U1 --> Y[音频后处理]
    
    T -->|静音| V[静音处理]
    V --> V1[音频片段删除]
    V1 --> Y
    
    T -->|语音转换| W[语音转换]
    W --> W1[Seed-VC模型推理]
    W1 --> W2[目标声音合成]
    W2 --> W3[音频片段替换]
    W3 --> Y
    
    T -->|音调变换| X[音调处理]
    X --> X1[音高/音色调整]
    X1 --> Y
    
    Y --> Z[淡入淡出处理]
    Z --> AA[音频质量优化]
    AA --> BB[生成最终音频]
    BB --> CC[结果验证]
    CC --> DD[输出匿名化音频]
    
    style A fill:#e1f5fe
    style DD fill:#c8e6c9
    style H fill:#fff3e0
    style L fill:#fff3e0
    style P1 fill:#fff3e0
    style W1 fill:#fff3e0
```

### 文本处理序列图

```mermaid
sequenceDiagram
    participant User as 用户
    participant UI as Gradio UI
    participant TP as TextProcessor
    participant SEG as 文本分割器
    participant PRES as Presidio分析器
    participant LLM as LLM识别器
    participant ANO as Presidio匿名化器
    participant CACHE as 缓存系统

    User->>UI: 输入文本内容
    UI->>TP: 处理文本请求
    
    TP->>TP: 文本预处理
    TP->>SEG: 启动文本分割
    
    SEG->>SEG: 选择分割策略
    alt 句子分割
        SEG->>SEG: spaCy句子分割
    else 段落分割
        SEG->>SEG: 段落分隔符分割
    else 固定长度分割
        SEG->>SEG: 固定长度+重叠分割
    end
    SEG-->>TP: 返回文本片段
    
    TP->>CACHE: 检查缓存
    CACHE-->>TP: 缓存结果(如有)
    
    alt 缓存未命中
        loop 处理每个文本片段
            TP->>PRES: Presidio规则检测
            PRES->>PRES: NLP模型分析
            PRES->>PRES: 正则模式匹配
            PRES-->>TP: 返回检测结果
            
            par 并行LLM检测
                TP->>LLM: 启动LLM智能识别
                LLM->>LLM: 构建提示模板
                LLM->>LLM: Gemini模型推理
                LLM->>LLM: 结果解析验证
                LLM-->>TP: 返回LLM结果
            end
            
            TP->>TP: 融合检测结果
            TP->>TP: 去重和置信度筛选
        end
        
        TP->>CACHE: 存储结果到缓存
    end
    
    TP->>ANO: 启动匿名化处理
    
    loop 处理每个PII实体
        ANO->>ANO: 选择匿名化策略
        alt 替换策略
            ANO->>ANO: 用标签替换
        else 掩码策略
            ANO->>ANO: 部分字符掩码
        else 哈希策略
            ANO->>ANO: 生成一致性哈希
        else 删除策略
            ANO->>ANO: 完全删除实体
        end
    end
    
    ANO->>ANO: 重构完整文本
    ANO->>ANO: 结果验证
    ANO-->>TP: 返回匿名化文本
    
    TP->>TP: 生成处理报告
    TP-->>UI: 返回完整结果
    UI-->>User: 展示原文、检测结果和匿名化文本
    
    Note over User,CACHE: 支持批量处理和增量更新
```

### 文本处理流程图

```mermaid
flowchart TD
    A[输入文本] --> B[文本预处理]
    B --> C[文本分割]
    
    C --> D{选择分割策略}
    D --> D1[句子分割]
    D --> D2[段落分割]
    D --> D3[固定长度分割]
    D --> D4[自定义分割]
    
    D1 --> E[PII实体检测]
    D2 --> E
    D3 --> E
    D4 --> E
    
    E --> F{检测方法}
    F --> F1[Presidio规则检测]
    F --> F2[LLM智能检测]
    F --> F3[自定义模式检测]
    
    F1 --> G[实体识别结果]
    F2 --> G
    F3 --> G
    
    G --> H[PII实体匿名化]
    H --> I{匿名化策略}
    
    I --> I1[替换策略]
    I --> I2[掩码策略]
    I --> I3[哈希策略]
    I --> I4[删除策略]
    
    I1 --> J[生成匿名化文本]
    I2 --> J
    I3 --> J
    I4 --> J
    
    J --> K[结果验证]
    K --> L[输出最终结果]
```

## 🚀 安装说明

### 系统要求

- Python 3.8+
- CUDA 11.8+ (推荐，用于GPU加速)
- 内存: 8GB+ (推荐16GB+)
- 存储空间: 10GB+ (用于模型缓存)

### 环境准备

1. **克隆项目**
```bash
git clone https://github.com/your-username/all-pii-in-one.git
cd all-pii-in-one
```

2. **创建虚拟环境**
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows
```

3. **安装依赖**
```bash
pip install -r requirements.txt
```

### 模型下载

系统会在首次运行时自动下载所需模型，包括：

- **Whisper模型**: 语音识别模型
- **Pyannote模型**: 说话人分离模型  
- **Seed-VC模型**: 语音转换模型
- **Presidio模型**: NLP实体识别模型

### 配置API密钥

设置环境变量：

```bash
# Hugging Face Token (用于访问Pyannote模型)
export HF_TOKEN="your_huggingface_token"

# Google Gemini API Key (用于LLM识别)
export GEMINI_API_KEY="your_gemini_api_key"

# OpenAI API Key (可选)
export OPENAI_API_KEY="your_openai_api_key"
```

## 🎮 快速开始

### 启动Web界面

```bash
python src/app/main.py
```

访问 `http://localhost:7860` 开始使用。

### 基本使用流程

1. **选择处理类型**: 文本、音频、图像或视频
2. **上传文件**: 拖拽或点击上传目标文件
3. **配置参数**: 根据需要调整处理参数
4. **开始处理**: 点击处理按钮启动分析
5. **查看结果**: 查看检测到的PII实体和匿名化结果
6. **下载文件**: 下载处理后的匿名化文件

### 命令行使用

```python
from src.processors.text_processor import TextProcessor
from src.configs import AppConfig

# 初始化
config = AppConfig.from_yaml_file("config/app_config.yaml")
processor = TextProcessor(config.processor.text_processor)

# 处理文本
text = "我的姓名是张三，电话号码是13812345678"
result = processor.process(text)

print(f"原文: {result.original_text}")
print(f"匿名化结果: {result.anonymized_text}")
```

## ⚙️ 配置说明

### 主配置文件

配置文件位于 `config/app_config.yaml`，包含以下主要部分：

#### 通用配置
```yaml
general:
  log_level: "INFO"
  temp_file_dir: "/tmp/pii_app"
  device: "cpu"  # 或 "cuda"
  ui:
    theme: "default"
    server_port: 7860
```

#### 文本处理器配置
```yaml
processor:
  text_processor:
    supported_entities:
      - PERSON
      - PHONE_NUMBER
      - EMAIL_ADDRESS
      - ID_CARD
    analyzer:
      presidio_enabled: true
      supported_languages: ["zh", "en"]
      enable_llm_recognizers: true
```

#### 音频处理器配置
```yaml
processor:
  audio_processor:
    transcription:
      model_size: "base"
      language: null  # 自动检测
    diarization:
      enabled: true
      min_speakers: 1
      max_speakers: 5
```

### 自定义配置

您可以通过以下方式自定义配置：

1. **修改配置文件**: 直接编辑 `config/app_config.yaml`
2. **环境变量覆盖**: 使用环境变量覆盖特定配置
3. **代码配置**: 在代码中动态设置配置参数

## 📚 使用指南

### 文本处理

#### 支持的PII实体类型

| 实体类型 | 描述 | 示例 |
|---------|------|------|
| PERSON | 人名 | 张三, 李四 |
| PHONE_NUMBER | 电话号码 | 13812345678 |
| EMAIL_ADDRESS | 邮箱地址 | user@example.com |
| ID_CARD | 身份证号 | 110101199001011234 |
| BANK_CARD | 银行卡号 | 6212261234567890123 |
| ADDRESS | 地址信息 | 北京市朝阳区xxx街道 |
| DATE_TIME | 日期时间 | 2024-01-01 |

#### 匿名化策略

1. **替换策略**: 用预定义标签替换
   - `张三` → `[PERSON]`
   - `13812345678` → `[PHONE]`

2. **掩码策略**: 部分字符掩码
   - `13812345678` → `138****5678`
   - `user@example.com` → `u***@example.com`

3. **哈希策略**: 生成一致性哈希
   - `张三` → `person_abc123`

### 音频处理

#### 支持的音频格式

- WAV (推荐)
- MP3
- FLAC
- OGG

#### 处理功能

1. **语音转录**: 将音频转换为文本
2. **说话人分离**: 识别不同说话人的时间段
3. **PII检测**: 在转录文本中检测PII实体
4. **音频匿名化**: 
   - 蜂鸣音替换
   - 静音处理
   - 语音转换

#### 最佳实践

- **音频质量**: 推荐使用采样率22050Hz或以上的音频
- **文件大小**: 支持大文件，但建议分段处理超长音频
- **语言设置**: 指定语言可提高转录准确性

## 🔧 API文档

### 文本处理API

```python
class TextProcessor:
    def process(self, text: str, 
                entities: List[str] = None,
                anonymization_strategy: str = "replace") -> ProcessingResult:
        """
        处理文本中的PII实体
        
        参数:
            text: 输入文本
            entities: 要检测的实体类型列表
            anonymization_strategy: 匿名化策略
            
        返回:
            ProcessingResult: 处理结果
        """
```

### 音频处理API

```python
class AudioProcessor:
    def process_audio(self, audio_path: str,
                     enable_diarization: bool = True,
                     enable_pii_detection: bool = True,
                     anonymization_method: str = "beep") -> AudioProcessingResult:
        """
        处理音频文件
        
        参数:
            audio_path: 音频文件路径
            enable_diarization: 是否启用说话人分离
            enable_pii_detection: 是否启用PII检测
            anonymization_method: 匿名化方法
            
        返回:
            AudioProcessingResult: 处理结果
        """
```

## 🛠️ 开发指南

### 项目结构

```
all-pii-in-one/
├── config/                 # 配置文件
│   ├── app_config.yaml     # 主配置文件
│   └── prompt_template.yaml # LLM提示模板
├── src/
│   ├── app/                # 应用层
│   │   ├── main.py         # 主应用入口
│   │   └── tabs/           # UI标签页
│   ├── processors/         # 处理器层
│   │   ├── text_processor/ # 文本处理器
│   │   ├── audio_processor/# 音频处理器
│   │   ├── image_processor/# 图像处理器
│   │   └── video_processor/# 视频处理器
│   ├── configs/            # 配置类
│   ├── commons/            # 公共工具
│   └── modules/            # 第三方模块
├── test/                   # 测试文件
├── data/                   # 数据文件
└── requirements.txt        # 依赖清单
```

### 添加新的处理器

1. **创建处理器类**
```python
from src.processors.base_processor import BaseProcessor

class CustomProcessor(BaseProcessor):
    def process(self, input_data):
        # 实现处理逻辑
        pass
```

2. **添加配置**
```yaml
processor:
  custom_processor:
    param1: value1
    param2: value2
```

3. **集成到UI**
```python
def create_custom_tab():
    with gr.TabItem("自定义处理"):
        # 创建UI组件
        pass
```

### 扩展PII实体类型

1. **添加识别规则**
```python
from presidio_analyzer import Pattern, PatternRecognizer

class CustomRecognizer(PatternRecognizer):
    PATTERNS = [
        Pattern("CUSTOM_ENTITY", r"regex_pattern", 0.8)
    ]
```

2. **配置匿名化策略**
```yaml
text_processor:
  anonymizer:
    entity_anonymization_config:
      CUSTOM_ENTITY:
        operator: "replace"
        params:
          new_value: "[CUSTOM]"
```

## ❓ 常见问题

### Q: 模型下载失败怎么办？

A: 检查网络连接，确保有足够的存储空间。某些模型需要Hugging Face token。

### Q: GPU内存不足怎么处理？

A: 在配置文件中设置 `device: "cpu"` 或减小 `batch_size`。

### Q: 支持哪些语言？

A: 目前主要支持中文和英文，可通过配置文件扩展其他语言。

### Q: 如何提高处理速度？

A: 
- 使用GPU加速
- 增加batch_size
- 启用缓存功能
- 使用更小的模型

### Q: 匿名化结果不准确怎么办？

A: 
- 调整实体检测阈值
- 添加自定义识别规则
- 使用LLM识别器
- 检查语言设置

## 🤝 贡献指南

我们欢迎社区贡献！请遵循以下步骤：

1. **Fork项目**
2. **创建特性分支**
```bash
git checkout -b feature/your-feature-name
```

3. **提交更改**
```bash
git commit -m "Add your feature"
```

4. **推送分支**
```bash
git push origin feature/your-feature-name
```

5. **创建Pull Request**

### 贡献类型

- 🐛 Bug修复
- ✨ 新功能
- 📚 文档改进
- 🎨 UI/UX改进
- ⚡ 性能优化
- 🧪 测试用例

### 代码规范

- 遵循PEP 8编码规范
- 添加适当的注释和文档字符串
- 编写单元测试
- 更新相关文档

## 📄 许可证

本项目采用 MIT 许可证。详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

感谢以下开源项目的贡献：

- [Presidio](https://github.com/microsoft/presidio) - PII检测和匿名化
- [WhisperX](https://github.com/m-bain/whisperX) - 语音识别和对齐
- [Pyannote](https://github.com/pyannote/pyannote-audio) - 说话人分离
- [Seed-VC](https://github.com/Plachtaa/seed-vc) - 语音转换
- [Gradio](https://github.com/gradio-app/gradio) - Web界面框架

## 📞 联系我们

- 📧 邮箱: 
- 🐛 问题反馈: [GitHub Issues](https://github.com/your-username/all-pii-in-one/issues)
- 💬 讨论: [GitHub Discussions](https://github.com/your-username/all-pii-in-one/discussions)

---

<div align="center">

**如果这个项目对您有帮助，请给我们一个 ⭐ Star！**

Made with ❤️ by the ALL PII IN ONE Team

</div>