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
    enabled: bool = Field(default=True, description="是否启用LLM识别器")
    supported_entities: list[str] = Field(
        default_factory=lambda: [
            "ALL"
        ],
        description="支持的实体类型列表"
    )
    select_prompt_template_name: str = Field(
        default="pii_detection",
        description="选择的提示词模板名称，默认为'pii_detection'"
    )
    client: LLMClientConfig = Field(default_factory=LLMClientConfig, description="LLM客户端配置")
    prompts: LLMPromptsConfig = Field(default_factory=LLMPromptsConfig, description="LLM提示词配置")
    parsers: LLMParsersConfig = Field(default_factory=LLMParsersConfig, description="LLM解析器配置")

__all__ = [
    "LLMRecognizerConfig",
    "LLMClientConfig",
    "LLMPromptsConfig",
    "LLMParsersConfig"
]
