#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器工厂模块

提供类型安全的解析器创建和管理功能，支持动态注册不同类型的解析器。
"""

from typing import Dict, Type, Optional, List, Union

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
from src.processors.text_processor.recognizers.llm.parsers.base_parser import BaseLLMParser

logger = get_module_logger(__name__)

# 全局解析器注册表
_parser_registry: Dict[str, Type[BaseLLMParser]] = {}
_parser_aliases: Dict[str, str] = {}


def register_parser(
    name: str, 
    parser_class: Type[BaseLLMParser],
    aliases: Optional[Union[str, List[str]]] = None
) -> None:
    """
    注册解析器类型
    
    Args:
        name: 解析器名称（唯一标识符）
        parser_class: 解析器类，必须继承自BaseLLMParser
        aliases: 解析器别名，可以是单个字符串或字符串列表
        
    Raises:
        ValueError: 当解析器名称已存在或解析器类不合法时
    """
    if name in _parser_registry:
        raise ValueError(f"解析器 '{name}' 已经注册")
        
    # 验证解析器类是否继承自BaseLLMParser
    if not issubclass(parser_class, BaseLLMParser):
        raise ValueError(f"解析器类 {parser_class.__name__} 必须继承自 BaseLLMParser")
        
    _parser_registry[name] = parser_class
    logger.debug(f"已注册LLM解析器: {name} -> {parser_class.__name__}")
    
    # 注册别名
    if aliases:
        if isinstance(aliases, str):
            aliases = [aliases]
        for alias in aliases:
            if alias in _parser_aliases:
                logger.warning(f"别名 '{alias}' 已存在，将被覆盖")
            _parser_aliases[alias] = name
            logger.debug(f"已注册解析器别名: {alias} -> {name}")


def unregister_parser(name: str) -> None:
    """
    注销解析器类型
    
    Args:
        name: 要注销的解析器名称
        
    Raises:
        KeyError: 当解析器名称不存在时
    """
    if name not in _parser_registry:
        raise KeyError(f"解析器 '{name}' 未注册")
        
    del _parser_registry[name]
    
    # 删除相关别名
    aliases_to_remove = [alias for alias, target in _parser_aliases.items() if target == name]
    for alias in aliases_to_remove:
        del _parser_aliases[alias]
        
    logger.debug(f"已注销LLM解析器: {name}")


def get_registered_parsers() -> Dict[str, Type[BaseLLMParser]]:
    """
    获取所有已注册的解析器类型
    
    Returns:
        Dict[str, Type[BaseLLMParser]]: 解析器名称到解析器类的映射
    """
    return _parser_registry.copy()


def get_parser_aliases() -> Dict[str, str]:
    """
    获取所有解析器别名
    
    Returns:
        Dict[str, str]: 别名到解析器名称的映射
    """
    return _parser_aliases.copy()


def is_parser_registered(name: str) -> bool:
    """
    检查解析器是否已注册
    
    Args:
        name: 解析器名称或别名
        
    Returns:
        bool: 如果解析器已注册返回True，否则返回False
    """
    return name in _parser_registry or name in _parser_aliases


def _resolve_parser_name(name: str) -> str:
    """
    解析解析器名称，处理别名
    
    Args:
        name: 解析器名称或别名
        
    Returns:
        str: 解析后的解析器名称
    """
    return _parser_aliases.get(name, name)


def create_parser(
    parser_type: str,
    config: Optional[LLMParsersConfig] = None
) -> BaseLLMParser:
    """
    根据类型创建解析器实例
    
    Args:
        parser_type: 解析器类型名称
        config: 解析器配置，如果为None则使用默认配置
        
    Returns:
        BaseLLMParser: 创建的解析器实例
        
    Raises:
        ValueError: 当解析器类型未注册时
        Exception: 当解析器创建失败时
    """
    if not parser_type:
        raise ValueError("解析器类型不能为空")
    
    # 如果没有提供配置，使用默认配置
    if config is None:
        config = LLMParsersConfig()
    
    # 解析解析器名称（处理别名）
    parser_name = _resolve_parser_name(parser_type)
    
    if parser_name not in _parser_registry:
        available_parsers = list(_parser_registry.keys())
        available_aliases = list(_parser_aliases.keys())
        raise ValueError(
            f"未知的解析器类型: '{parser_type}'。"
            f"可用的解析器类型: {available_parsers}，"
            f"可用的别名: {available_aliases}"
        )
    
    parser_class = _parser_registry[parser_name]
    
    try:
        # 创建解析器实例
        parser_instance = parser_class(config)
        logger.info(f"成功创建LLM解析器: {parser_name} ({parser_class.__name__})")
        return parser_instance
        
    except Exception as e:
        logger.error(f"创建LLM解析器失败: {parser_name} - {str(e)}")
        raise Exception(f"创建解析器 '{parser_name}' 失败: {str(e)}") from e


def get_parser_class(name: str) -> Type[BaseLLMParser]:
    """
    获取指定名称的解析器类
    
    Args:
        name: 解析器名称或别名
        
    Returns:
        Type[BaseLLMParser]: 解析器类
        
    Raises:
        KeyError: 当解析器未注册时
    """
    parser_name = _resolve_parser_name(name)
    if parser_name not in _parser_registry:
        raise KeyError(f"解析器 '{name}' 未注册")
    return _parser_registry[parser_name]


def _register_builtin_parsers() -> None:
    """注册内置的解析器类型"""
    try:
        # 注册Gemini解析器
        from .gemini_parser import GeminiParser
        register_parser("gemini", GeminiParser, ["google-gemini", "google"])
        
        # 可以在这里注册其他解析器
        # register_parser("openai", OpenAIParser, ["gpt", "chatgpt"])
        # register_parser("claude", ClaudeParser, ["anthropic"])
        
    except ImportError as e:
        logger.warning(f"无法导入内置解析器: {e}")
    
    logger.info(f"已注册 {len(get_registered_parsers())} 个LLM解析器类型")


# 在模块加载时自动注册内置解析器
_register_builtin_parsers()
