#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本分割器配置类
"""

from typing import List, Optional
from pydantic import Field

from src.configs.base_config import BaseConfig


class SegmentationConfig(BaseConfig):
    """文本分割器配置"""
    
    # 分割策略
    segmentation_strategy: str = Field(
        default="sentence",
        description="分割策略 (sentence, paragraph, fixed_length, custom)"
    )
    
    # 句子分割配置
    sentence_segmentation: dict = Field(
        default_factory=lambda: {
            "use_spacy": True,
            "spacy_model": "zh_core_web_sm",
            "custom_patterns": [r"[。！？]", r"[.!?]"],
            "min_sentence_length": 5,
            "max_sentence_length": 1000
        },
        description="句子分割配置"
    )
    
    # 段落分割配置
    paragraph_segmentation: dict = Field(
        default_factory=lambda: {
            "paragraph_separators": ["\n\n", "\r\n\r\n"],
            "min_paragraph_length": 10,
            "max_paragraph_length": 5000
        },
        description="段落分割配置"
    )
    
    # 固定长度分割配置
    fixed_length_segmentation: dict = Field(
        default_factory=lambda: {
            "chunk_size": 500,
            "overlap_size": 50,
            "respect_word_boundaries": True
        },
        description="固定长度分割配置"
    )
    
    # 自定义分割配置
    custom_segmentation: dict = Field(
        default_factory=lambda: {
            "custom_patterns": [],
            "custom_function": None
        },
        description="自定义分割配置"
    )
    
    # 后处理配置
    post_processing: dict = Field(
        default_factory=lambda: {
            "remove_empty_segments": True,
            "strip_whitespace": True,
            "min_segment_length": 1,
            "max_segment_length": 10000
        },
        description="后处理配置"
    )
    
    # 性能配置
    parallel_processing: bool = Field(
        default=False,
        description="是否启用并行处理"
    )
    
    # 缓存配置
    enable_cache: bool = Field(
        default=True,
        description="是否启用分割结果缓存"
    )
    cache_size: int = Field(
        default=500,
        description="缓存大小",
        ge=0
    )