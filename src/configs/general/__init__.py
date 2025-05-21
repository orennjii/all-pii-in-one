#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
通用配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig
from src.configs.general.ui import UIConfig


class GeneralConfig(BaseConfig):
    """通用配置"""
    log_level: str = Field(default="INFO", description="日志级别")
    temp_file_dir: str = Field(default="/tmp/pii_app", description="临时文件存储目录")
    ui: UIConfig = Field(default_factory=UIConfig, description="UI相关设置")
