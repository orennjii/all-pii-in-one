#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
UI配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class UIConfig(BaseConfig):
    """UI配置"""
    theme: str = Field(default="default", description="UI主题")
    share_ui: bool = Field(default=False, description="是否通过Gradio分享UI")
    server_name: str = Field(default="0.0.0.0", description="服务器名称")
    server_port: int = Field(default=7860, description="服务器端口")
