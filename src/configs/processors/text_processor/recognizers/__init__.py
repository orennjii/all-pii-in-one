#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
识别器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from .pattern import PatternRecognizerConfig
from .llm import LLMRecognizerConfig


class RecognizersConfig(BaseConfig):
    """识别器配置"""
    pattern: PatternRecognizerConfig = Field(default_factory=PatternRecognizerConfig, description="模式识别器配置")
    llm: LLMRecognizerConfig = Field(default_factory=LLMRecognizerConfig, description="LLM识别器配置")

__all__ = [
    "RecognizersConfig",
    "PatternRecognizerConfig",
    "LLMRecognizerConfig"
]
