#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器模块

提供用于解析不同LLM响应的解析器实现，将LLM响应转换为标准的PII识别结果格式。
"""

from src.processors.text_processor.recognizers.llm.parsers.parser import ResponseParser
from src.processors.text_processor.recognizers.llm.parsers.gemini_parser import GeminiResponseParser
from src.processors.text_processor.recognizers.llm.parsers.parser_factory import create_response_parser

__all__ = [
    'ResponseParser',
    'GeminiResponseParser',
    'create_response_parser',
]