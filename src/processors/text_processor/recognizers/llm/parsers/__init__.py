#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器模块

提供用于解析LLM响应的各种解析器，将LLM的原始输出转换为标准的实体识别结果。
"""

from .entity_match import EntityMatch
from .base_parser import BaseLLMParser, LLMResponse
from .gemini_parser import GeminiParser
from .parser_factory import (
    create_parser,
    register_parser,
    unregister_parser,
    get_registered_parsers,
    get_parser_aliases,
    is_parser_registered,
    get_parser_class
)

__all__ = [
    # 数据结构
    "EntityMatch",
    "LLMResponse",
    
    # 解析器类
    "BaseLLMParser",
    "GeminiParser",
    
    # 工厂函数
    "create_parser",
    "register_parser",
    "unregister_parser",
    "get_registered_parsers",
    "get_parser_aliases",
    "is_parser_registered",
    "get_parser_class",
]