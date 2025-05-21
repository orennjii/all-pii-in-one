#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from src.configs.processors.text_processor.recognizers import RecognizersConfig


class TextProcessorConfig(BaseConfig):
    """文本处理器配置"""
    recognizers: RecognizersConfig = Field(default_factory=RecognizersConfig, description="识别器配置")
    # 可以在这里添加其他文本处理器相关的配置

__all__ = [
    "RecognizersConfig",
    "TextProcessorConfig"
]