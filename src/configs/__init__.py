"""
配置模块包
提供各种服务的配置类和默认配置
"""

from .base_config import BaseConfig
from .audio_config import (
    AudioConfig,
    AudioDiarizationConfig,
    AudioVoiceConversionConfig,
    AudioSupportedFormatsConfig,
    # 导出全局单例实例
    AUDIO_CONFIG,
)

__all__ = [
    "BaseConfig",
    "AudioConfig",
    "AudioDiarizationConfig",
    "AudioVoiceConversionConfig",
    "AudioSupportedFormatsConfig",
    "AUDIO_CONFIG",
]