#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM客户端配置模块
"""

from typing import Optional
from pydantic import Field

from src.configs.base_config import BaseConfig


class LLMClientConfig(BaseConfig):
    """LLM客户端配置"""
    type: str = Field(default="HuggingFace", description="LLM客户端类型")
    model_name_or_path: str = Field(default="meta-llama/Llama-2-7b-chat-hf", description="模型名称或路径")
    temperatures: float = Field(default=0.7, description="LLM温度参数")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    endpoint: Optional[str] = Field(default=None, description="API端点")
