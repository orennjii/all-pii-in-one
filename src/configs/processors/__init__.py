#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
处理器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from .text_processor import TextProcessorConfig
from .audio_processor.audio_configs import AudioProcessorConfig

class ProcessorsConfig(BaseConfig):
    """处理器配置"""
    
    audio_processor: AudioProcessorConfig = Field(
        default_factory=AudioProcessorConfig,
        description="音频处理器配置"
    )
    text_processor: TextProcessorConfig = Field(
        default_factory=TextProcessorConfig, 
        description="文本处理器配置"
    )

__all__ = [
    "ProcessorsConfig",
    "AudioProcessorConfig",
    "TextProcessorConfig"
]
