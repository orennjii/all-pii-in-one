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
    type: str = Field(default="gemini", description="LLM客户端类型")
    model_name_or_path: str = Field(default="gemini-2.5-flash-preview-05-20", description="模型名称或路径")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    endpoint: Optional[str] = Field(default=None, description="API端点")

    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="系统提示词，用于引导模型的行为"
    )

    temperatures: float = Field(default=0.7, description="LLM温度参数")
    max_tokens: int = Field(default=20000, description="LLM最大token数")
    top_p: float = Field(default=0.9, description="LLM采样参数")
    top_k: int = Field(default=50, description="LLM采样参数")

    # google 特有参数
    use_vertex: bool = Field(default=False, description="是否使用Vertex AI")
    project_id: Optional[str] = Field(
        default=None,
        description="Google Cloud项目ID, 如果使用Vertex AI, 则需要提供此参数"
    )
    location: Optional[str] = Field(
        default=None,
        description="Google Cloud区域, 如果使用Vertex AI, 则需要提供此参数"
    )
