#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理器核心模块

此模块提供文本处理的核心功能，包括：
1. PII分析器 - 检测文本中的敏感信息
2. PII匿名化器 - 对检测到的敏感信息进行匿名化处理
3. 文本分割器 - 将文本分割为合适的处理单元

所有模块都遵循面向接口编程的原则，支持灵活的扩展和配置。
"""

from .analyzer import PresidioAnalyzer
from .anonymizer import PresidioAnonymizer
from .segmentation import TextSegmenter, TextSegment

__all__ = [
    # 分析器
    "PresidioAnalyzer",
    
    # 匿名化器
    "PresidioAnonymizer",
    
    # 分割器
    "TextSegmenter",
    "TextSegment",
]