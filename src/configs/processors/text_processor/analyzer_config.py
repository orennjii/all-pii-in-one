#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PII分析器配置类
"""

from typing import List, Optional
from pydantic import Field

from src.configs.base_config import BaseConfig


class AnalyzerConfig(BaseConfig):
    """PII分析器配置"""
    
    # Presidio Analyzer 配置
    presidio_enabled: bool = Field(default=True, description="是否启用Presidio分析器")
    supported_languages: List[str] = Field(
        default=["zh", "en"], 
        description="支持的语言列表"
    )
    default_language: str = Field(default="zh", description="默认语言")
    
    # 识别器配置
    enable_pattern_recognizers: bool = Field(
        default=True, 
        description="是否启用模式识别器"
    )
    enable_llm_recognizers: bool = Field(
        default=True, 
        description="是否启用LLM识别器"
    )
    
    # 分析阈值
    default_score_threshold: float = Field(
        default=0.5, 
        description="默认置信度阈值",
        ge=0.0,
        le=1.0
    )
    
    # 实体类型过滤
    allowed_entities: Optional[List[str]] = Field(
        default=None,
        description="允许的实体类型列表，None表示允许所有类型"
    )
    denied_entities: Optional[List[str]] = Field(
        default=None,
        description="拒绝的实体类型列表"
    )
    
    # 性能配置
    parallel_processing: bool = Field(
        default=False,
        description="是否启用并行处理"
    )
    max_workers: int = Field(
        default=4,
        description="最大工作线程数",
        ge=1,
        le=16
    )
    
    # 缓存配置
    enable_cache: bool = Field(
        default=True,
        description="是否启用结果缓存"
    )
    cache_size: int = Field(
        default=1000,
        description="缓存大小",
        ge=0
    )