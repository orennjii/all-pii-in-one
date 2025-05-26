"""
配置模块包
提供各种服务的配置类和默认配置
"""

from .base_config import ConfigType, BaseConfig
from .general import GeneralConfig
from .device import DeviceConfig
from .processors import ProcessorsConfig
from .processors.audio_processor import AudioProcessorConfig
from .processors.text_processor import TextProcessorConfig
from .processors.text_processor.recognizers import (
    RecognizersConfig,
    LLMRecognizerConfig,
    PatternRecognizerConfig,
)
from .processors.text_processor.recognizers.llm import (
    LLMClientConfig,
    LLMPromptsConfig,
    LLMParsersConfig,
)
from .processors.text_processor.recognizers.pattern import (
    PatternRecognizerConfig,
)

# 创建主应用配置类
from pydantic import Field
class AppConfig(BaseConfig):
    """应用程序主配置"""
    general: GeneralConfig = Field(default_factory=GeneralConfig, description="通用配置")
    device: DeviceConfig = Field(default_factory=DeviceConfig, description="设备配置")
    processor: ProcessorsConfig = Field(default_factory=ProcessorsConfig, description="处理器配置")
    # 可以在这里添加其他处理器的配置，如图像处理器等

__all__ = [
    "ConfigType",
    "BaseConfig",
    
    # 新的配置结构
    "GeneralConfig",
    "DeviceConfig",
    "ProcessorsConfig",
    "AudioProcessorConfig",
    "TextProcessorConfig",
    "AppConfig"
]