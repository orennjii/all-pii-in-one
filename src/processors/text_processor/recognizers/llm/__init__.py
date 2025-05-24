#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM识别器模块

该模块提供基于大语言模型的PII识别能力，通过调用大语言模型API来识别文本中的个人隐私信息。
"""

from src.processors.text_processor.recognizers.llm.recognizer import LLMRecognizer

__all__ = [
    "LLMRecognizer",
]