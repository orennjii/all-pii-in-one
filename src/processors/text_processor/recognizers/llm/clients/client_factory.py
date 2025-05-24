#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM客户端工厂模块

提供类型安全的LLM客户端创建和管理功能，使用泛型确保类型安全。
支持动态注册不同类型的LLM客户端，并根据配置创建相应的客户端实例。
"""

from typing import Dict, Type, TypeVar, Generic, Optional, Union, List, cast
from abc import ABC

from src.configs.processors.text_processor.recognizers.llm import LLMClientConfig
from src.commons.loggers import get_module_logger
from src.processors.text_processor.recognizers.llm.clients.base_client import BaseLLMClient

logger = get_module_logger(__name__)

# 定义泛型类型变量，绑定到BaseLLMClient
ClientType = TypeVar('ClientType', bound=BaseLLMClient)


class LLMClientFactory(Generic[ClientType]):
    """
    LLM客户端工厂类
    
    使用泛型提供类型安全的客户端注册和创建功能。
    支持动态注册不同类型的LLM客户端，并根据配置创建相应的客户端实例。
    """
    
    def __init__(self) -> None:
        """初始化客户端工厂"""
        self._clients: Dict[str, Type[ClientType]] = {}
        self._aliases: Dict[str, str] = {}
        
    def register_client(
        self, 
        name: str, 
        client_class: Type[ClientType],
        aliases: Optional[Union[str, List[str]]] = None
    ) -> None:
        """
        注册LLM客户端类型
        
        Args:
            name: 客户端名称（唯一标识符）
            client_class: 客户端类，必须继承自BaseLLMClient
            aliases: 客户端别名，可以是单个字符串或字符串列表
            
        Raises:
            ValueError: 当客户端名称已存在或客户端类不合法时
        """
        if name in self._clients:
            raise ValueError(f"客户端 '{name}' 已经注册")
            
        # 验证客户端类是否继承自BaseLLMClient
        if not issubclass(client_class, BaseLLMClient):
            raise ValueError(f"客户端类 {client_class.__name__} 必须继承自 BaseLLMClient")
            
        self._clients[name] = client_class
        logger.debug(f"已注册LLM客户端: {name} -> {client_class.__name__}")
        
        # 注册别名
        if aliases:
            if isinstance(aliases, str):
                aliases = [aliases]
            for alias in aliases:
                if alias in self._aliases:
                    logger.warning(f"别名 '{alias}' 已存在，将被覆盖")
                self._aliases[alias] = name
                logger.debug(f"已注册客户端别名: {alias} -> {name}")
    
    def unregister_client(self, name: str) -> None:
        """
        注销LLM客户端类型
        
        Args:
            name: 要注销的客户端名称
            
        Raises:
            KeyError: 当客户端名称不存在时
        """
        if name not in self._clients:
            raise KeyError(f"客户端 '{name}' 未注册")
            
        del self._clients[name]
        
        # 删除相关别名
        aliases_to_remove = [alias for alias, target in self._aliases.items() if target == name]
        for alias in aliases_to_remove:
            del self._aliases[alias]
            
        logger.debug(f"已注销LLM客户端: {name}")
    
    def get_registered_clients(self) -> Dict[str, Type[ClientType]]:
        """
        获取所有已注册的客户端类型
        
        Returns:
            Dict[str, Type[ClientType]]: 客户端名称到客户端类的映射
        """
        return self._clients.copy()
    
    def get_client_aliases(self) -> Dict[str, str]:
        """
        获取所有客户端别名
        
        Returns:
            Dict[str, str]: 别名到客户端名称的映射
        """
        return self._aliases.copy()
    
    def is_registered(self, name: str) -> bool:
        """
        检查客户端是否已注册
        
        Args:
            name: 客户端名称或别名
            
        Returns:
            bool: 如果客户端已注册返回True，否则返回False
        """
        return name in self._clients or name in self._aliases
    
    def _resolve_client_name(self, name: str) -> str:
        """
        解析客户端名称，处理别名
        
        Args:
            name: 客户端名称或别名
            
        Returns:
            str: 解析后的客户端名称
        """
        return self._aliases.get(name, name)
    
    def create_client(
        self, 
        config: LLMClientConfig
    ) -> ClientType:
        """
        根据配置创建LLM客户端实例
        
        Args:
            config: LLM客户端配置
            
        Returns:
            ClientType: 创建的客户端实例
            
        Raises:
            ValueError: 当客户端类型未注册或配置无效时
            Exception: 当客户端创建失败时
        """
        if not config.type:
            raise ValueError("配置中未指定客户端类型")
            
        # 解析客户端名称（处理别名）
        client_name = self._resolve_client_name(config.type)
        
        if client_name not in self._clients:
            available_clients = list(self._clients.keys())
            available_aliases = list(self._aliases.keys())
            raise ValueError(
                f"未知的客户端类型: '{config.type}'。"
                f"可用的客户端类型: {available_clients}，"
                f"可用的别名: {available_aliases}"
            )
        
        client_class = self._clients[client_name]
        
        try:
            # 创建客户端实例
            client_instance = client_class(config)
            logger.info(f"成功创建LLM客户端: {client_name} ({client_class.__name__})")
            return client_instance
            
        except Exception as e:
            logger.error(f"创建LLM客户端失败: {client_name} - {str(e)}")
            raise Exception(f"创建客户端 '{client_name}' 失败: {str(e)}") from e
    
    def get_client_class(self, name: str) -> Type[ClientType]:
        """
        获取指定名称的客户端类
        
        Args:
            name: 客户端名称或别名
            
        Returns:
            Type[ClientType]: 客户端类
            
        Raises:
            KeyError: 当客户端未注册时
        """
        client_name = self._resolve_client_name(name)
        if client_name not in self._clients:
            raise KeyError(f"客户端 '{name}' 未注册")
        return self._clients[client_name]


# 创建全局工厂实例
_client_factory: LLMClientFactory[BaseLLMClient] = LLMClientFactory()


def register_client(
    name: str, 
    client_class: Type[BaseLLMClient],
    aliases: Optional[Union[str, List[str]]] = None
) -> None:
    """
    注册LLM客户端类型到全局工厂
    
    Args:
        name: 客户端名称
        client_class: 客户端类
        aliases: 客户端别名
    """
    _client_factory.register_client(name, client_class, aliases)


def unregister_client(name: str) -> None:
    """
    从全局工厂注销LLM客户端类型
    
    Args:
        name: 客户端名称
    """
    _client_factory.unregister_client(name)


def get_registered_clients() -> Dict[str, Type[BaseLLMClient]]:
    """
    获取所有已注册的客户端类型
    
    Returns:
        Dict[str, Type[BaseLLMClient]]: 客户端名称到客户端类的映射
    """
    return _client_factory.get_registered_clients()


def is_client_registered(name: str) -> bool:
    """
    检查客户端是否已注册
    
    Args:
        name: 客户端名称或别名
        
    Returns:
        bool: 如果客户端已注册返回True，否则返回False
    """
    return _client_factory.is_registered(name)


def create_llm_client(config: LLMClientConfig) -> BaseLLMClient:
    """
    根据配置创建LLM客户端实例
    
    这是主要的工厂函数，用于根据配置创建相应的LLM客户端。
    
    Args:
        config: LLM客户端配置，包含客户端类型和其他参数
        
    Returns:
        BaseLLMClient: 创建的客户端实例
        
    Raises:
        ValueError: 当客户端类型未注册或配置无效时
        Exception: 当客户端创建失败时
        
    Example:
        >>> config = LLMClientConfig(client_type="gemini", api_key="xxx")
        >>> client = create_llm_client(config)
        >>> isinstance(client, BaseLLMClient)
        True
    """
    return _client_factory.create_client(config)


def get_client_factory() -> LLMClientFactory[BaseLLMClient]:
    """
    获取全局客户端工厂实例
    
    Returns:
        LLMClientFactory[BaseLLMClient]: 全局工厂实例
    """
    return _client_factory


# 自动注册已知的客户端类型
def _register_builtin_clients() -> None:
    """注册内置的客户端类型"""
    try:
        # 注册Gemini客户端
        from .gemini_client import GeminiClient
        register_client("gemini", GeminiClient, ["google-gemini", "google", "Gemini"])
        
    except ImportError as e:
        logger.warning(f"无法导入Gemini客户端: {e}")
    
    logger.info(f"已注册 {len(get_registered_clients())} 个LLM客户端类型")


# 在模块加载时自动注册内置客户端
_register_builtin_clients()
