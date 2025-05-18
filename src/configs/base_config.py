#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
基础配置类模块，提供共享的配置功能
"""

import os
import yaml
from functools import lru_cache
from typing import Any, Dict, Optional, Type, TypeVar, Generic, ClassVar
from pydantic import BaseModel, Field, ConfigDict


T = TypeVar('T', bound='BaseConfig')


class BaseConfig(BaseModel):
    """
    基础配置类，所有其他配置类应该继承这个类
    提供了从YAML文件加载配置的方法
    """
    
    model_config = ConfigDict(
        frozen=True,  # 使配置不可变
        populate_by_name=True,
        extra="ignore",  # 忽略额外字段
        arbitrary_types_allowed=True,
    )


    # 指定默认配置文件路径的类变量，子类应覆盖
    DEFAULT_CONFIG_PATH: ClassVar[str] = ""
    
    @classmethod
    def load_from_yaml(cls: Type[T], yaml_path: Optional[str] = None) -> T:
        """
        从YAML文件加载配置
        
        参数:
            yaml_path: YAML配置文件路径，如果为None则使用默认路径
            
        返回:
            配置对象
        """
        # 如果未指定路径，则使用默认路径
        config_path = yaml_path or cls.DEFAULT_CONFIG_PATH
        
        if not config_path:
            raise ValueError(f"{cls.__name__} 未指定配置文件路径")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 读取YAML文件
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
        
        # 创建配置实例
        return cls(**config_data)
    
    @classmethod
    @lru_cache(maxsize=1)
    def get_instance(cls: Type[T], yaml_path: Optional[str] = None) -> T:
        """
        获取配置的单例实例，使用缓存确保只创建一次
        
        参数:
            yaml_path: YAML配置文件路径，如果为None则使用默认路径
            
        返回:
            配置对象的单例实例
        """
        return cls.load_from_yaml(yaml_path)
