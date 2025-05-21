#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM提示词配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class LLMPromptsConfig(BaseConfig):
    """LLM提示词配置"""
    prompt_template_path: str = Field(
        default="src/processors/text_processor/recognizers/llm/prompts/default_prompt.json",
        description="提示词模板路径"
    )
