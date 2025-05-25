#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PII匿名化器配置类
"""

from typing import Dict, Any
from pydantic import Field

from src.configs.base_config import BaseConfig


class AnonymizerConfig(BaseConfig):
    """PII匿名化器配置 - 简化版本"""
    
    # Presidio Anonymizer 配置
    presidio_enabled: bool = Field(default=True, description="是否启用Presidio匿名化器")
    
    # 统一的匿名化策略
    anonymization_strategy: str = Field(
        default="replace",
        description="统一匿名化策略 (replace, mask, redact, hash, encrypt)"
    )
    
    # 统一的操作符参数
    operator_params: Dict[str, Any] = Field(
        default_factory=lambda: {
            "new_value": "[REDACTED]",  # 用于replace操作
            "masking_char": "*",        # 用于mask操作
            "chars_to_mask": -1,        # -1表示掩码整个实体
            "from_end": True,           # 从尾部开始掩码
            "hash_type": "sha256"       # 用于hash操作
        },
        description="操作符参数配置"
    )
    
    # 保留配置
    keep_original_score: bool = Field(
        default=False,
        description="是否在匿名化结果中保留原始置信度分数"
    )