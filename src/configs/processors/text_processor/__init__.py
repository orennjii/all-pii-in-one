#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from src.configs.processors.text_processor.analyzer_config import AnalyzerConfig
from src.configs.processors.text_processor.anonymizer_config import AnonymizerConfig
from src.configs.processors.text_processor.segmentation_config import SegmentationConfig
from src.configs.processors.text_processor.recognizers import RecognizersConfig

class TextProcessorConfig(BaseConfig):
    """文本处理器配置"""
    supported_entities: list[str] = Field(
        default_factory=lambda: [
            "PERSON", "LOCATION", "ORGANIZATION", "EMAIL_ADDRESS",
            "PHONE_NUMBER", "CREDIT_CARD", "DATE_TIME", "IP_ADDRESS",
            "BANK_ACCOUNT", "ID_CARD", "URL", "CURRENCY", "AGE",
        ],
        description="支持的实体类型列表"
    )
    exclude_entities: list[str] = Field(
        default_factory=lambda: ["NONE"],
        description="排除的实体类型列表"
    )
    recognizers: RecognizersConfig = Field(default_factory=RecognizersConfig, description="识别器配置")
    analyzer: AnalyzerConfig = Field(default_factory=AnalyzerConfig, description="分析器配置")
    anonymizer: AnonymizerConfig = Field(default_factory=AnonymizerConfig, description="匿名化配置")
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig, description="分词配置")

__all__ = [
    "RecognizersConfig",
    "AnalyzerConfig",
    "AnonymizerConfig",
    "SegmentationConfig",
    "TextProcessorConfig"
]