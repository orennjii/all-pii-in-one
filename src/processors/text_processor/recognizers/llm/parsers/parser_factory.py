#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器工厂模块

该模块提供创建不同类型LLM响应解析器的工厂函数。
"""

from typing import Optional, Dict, Any, Type

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
from src.processors.text_processor.recognizers.llm.parsers.parser import ResponseParser
from src.processors.text_processor.recognizers.llm.parsers.gemini_parser import GeminiResponseParser

logger = get_module_logger(__name__)

# 解析器类型映射
PARSER_TYPES = {
    "default": ResponseParser,  # 默认解析器，实际应用中可能需要一个具体实现
    "gemini": GeminiResponseParser,
}


def create_response_parser(
    parser_type: str = None,
    config: Optional[LLMParsersConfig] = None,
    **kwargs
) -> ResponseParser:
    """
    创建LLM响应解析器
    
    根据指定的类型或配置创建相应的解析器实例。
    
    Args:
        parser_type: 解析器类型，如果为None则从配置中读取
        config: LLM解析器配置
        **kwargs: 传递给解析器构造函数的额外参数
        
    Returns:
        ResponseParser: 创建的解析器实例
        
    Raises:
        ValueError: 如果指定的解析器类型不受支持
    """
    if config is None:
        from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
        config = LLMParsersConfig()
        
    # 如果没有指定解析器类型，从配置中读取
    if parser_type is None:
        parser_type = config.parser_type
        
    # 获取对应的解析器类
    parser_class = PARSER_TYPES.get(parser_type.lower())
    
    if not parser_class:
        available_types = ", ".join(PARSER_TYPES.keys())
        logger.error(f"不支持的解析器类型: {parser_type}，可用类型: {available_types}")
        raise ValueError(f"不支持的解析器类型: {parser_type}，可用类型: {available_types}")
    
    try:
        # 创建解析器实例
        parser = parser_class(config=config, **kwargs)
        logger.debug(f"已创建 {parser_class.__name__} 解析器")
        return parser
    except Exception as e:
        logger.error(f"创建解析器失败: {str(e)}")
        raise