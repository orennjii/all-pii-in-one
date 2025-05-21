#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM解析器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class LLMParsersConfig(BaseConfig):
    """LLM解析器配置"""
    example: str = Field(default="default", description="解析器示例配置")
