#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM识别器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from src.configs.processors.text_processor.recognizers.llm.client_config import LLMClientConfig
from src.configs.processors.text_processor.recognizers.llm.prompts_config import LLMPromptsConfig
from src.configs.processors.text_processor.recognizers.llm.parsers_config import LLMParsersConfig


class LLMRecognizerConfig(BaseConfig):
    """LLM识别器配置"""
    enabled: bool = Field(default=False, description="是否启用LLM识别器")
    client: LLMClientConfig = Field(default_factory=LLMClientConfig, description="LLM客户端配置")
    prompts: LLMPromptsConfig = Field(default_factory=LLMPromptsConfig, description="LLM提示词配置")
    parsers: LLMParsersConfig = Field(default_factory=LLMParsersConfig, description="LLM解析器配置")

__all__ = [
    "LLMRecognizerConfig",
    "LLMClientConfig",
    "LLMPromptsConfig",
    "LLMParsersConfig"
]
