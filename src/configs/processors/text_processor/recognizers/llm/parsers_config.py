#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM解析器配置模块
"""

from typing import Dict, Any, Optional, List
from pydantic import Field

from src.configs.base_config import BaseConfig


class LLMParsersConfig(BaseConfig):
    """LLM响应解析器配置"""
    
    parser_type: str = Field(
        default="default", 
        description="解析器类型，可选值: default, gemini"
    )
    
    default_parser: str = Field(
        default="gemini",
        description="默认使用的解析器类型"
    )
    
    # JSON解析配置
    json_strict_mode: bool = Field(
        default=False, 
        description="是否使用严格模式解析JSON"
    )
    
    # 通用配置
    min_confidence: float = Field(
        default=0.5, 
        description="最低置信度阈值"
    )
    
    # Gemini特定配置
    gemini_response_formats: List[str] = Field(
        default=["json", "table", "text"],
        description="支持的Gemini响应格式列表"
    )
