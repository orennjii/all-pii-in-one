#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础配置类模块，提供共享的配置功能，支持嵌套配置
"""

import os
import yaml
from functools import lru_cache
from typing import Any, Dict, Optional, Type, TypeVar, Generic, ClassVar, get_type_hints, get_origin, get_args
from pydantic import BaseModel, Field, ConfigDict, create_model


ConfigType = TypeVar('ConfigType', bound='BaseConfig')


class BaseConfig(BaseModel):
    """
    基础配置类，所有其他配置类应该继承这个类
    提供了从YAML文件加载配置的方法，支持嵌套配置
    """
    
    model_config = ConfigDict(
        frozen=True,  # 使配置不可变
        populate_by_name=True,
        extra="ignore",  # 忽略额外字段
        arbitrary_types_allowed=True,
    )
    
    @classmethod
    def from_dict(
        cls: Type[ConfigType], 
        cfg_dict: Dict[Any, Any],
    ) -> ConfigType:
        """
        从字典创建配置对象，支持嵌套配置类
        
        Args:
            cfg_dict (Dict[Any, Any]): 配置字典
        
        Returns:
            T: 配置类的实例化对象
        """
        # 处理嵌套配置
        processed_dict = {}
        type_hints = get_type_hints(cls)
        
        for key, value in cfg_dict.items():
            # 如果字段存在于类型提示中并且是嵌套的BaseConfig类型
            if key in type_hints:
                field_type = type_hints[key]
                origin_type = get_origin(field_type)
                
                # 如果字段类型是一个嵌套的BaseConfig
                if isinstance(value, dict) and hasattr(field_type, "__mro__") and BaseConfig in field_type.__mro__:
                    processed_dict[key] = field_type.from_dict(value)
                # 如果是Optional[SomeConfig]这样的类型
                elif origin_type is Optional:
                    args = get_args(field_type)
                    if args and isinstance(value, dict) and hasattr(args[0], "__mro__") and BaseConfig in args[0].__mro__:
                        processed_dict[key] = args[0].from_dict(value)
                    else:
                        processed_dict[key] = value
                else:
                    processed_dict[key] = value
            else:
                processed_dict[key] = value
                
        # 使用处理后的字典创建配置对象
        return cls(**processed_dict)
        

    @classmethod
    def load_from_yaml(cls: Type[ConfigType], yaml_path: Optional[str] = None) -> ConfigType:
        """
        从YAML文件加载配置，支持嵌套配置
        
        参数:
            yaml_path: YAML配置文件路径，如果为None则使用默认路径
            
        返回:
            配置对象
        """
        if not yaml_path:
            raise ValueError(f"{cls.__name__} 未指定配置文件路径")

        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"配置文件不存在: {yaml_path}")

        # 读取YAML文件
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 使用 from_dict 方法处理嵌套配置
        return cls.from_dict(config_data)
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls: Type[ConfigType], yaml_path: Optional[str] = None) -> ConfigType:
        """
        获取配置的单例实例，使用缓存确保只创建一次
        
        参数:
            yaml_path: YAML配置文件路径，如果为None则使用默认路径
            
        返回:
            配置对象的单例实例
        """
        return cls.load_from_yaml(yaml_path)
        
    def update(self, update_dict: Dict[str, Any]) -> ConfigType:
        """
        使用给定的字典更新当前配置，返回一个新的配置实例
        
        参数:
            update_dict: 包含要更新的配置的字典
            
        返回:
            更新后的新配置对象
        """
        # 将当前配置转换为字典
        current_dict = self.model_dump()
        
        # 递归更新字典
        def deep_update(original: Dict[str, Any], update: Dict[str, Any]) -> Dict[str, Any]:
            for key, value in update.items():
                if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                    original[key] = deep_update(original[key], value)
                else:
                    original[key] = value
            return original
        
        # 更新配置字典
        updated_dict = deep_update(current_dict, update_dict)
        
        # 使用更新后的字典创建新的配置对象
        return self.__class__.from_dict(updated_dict)
