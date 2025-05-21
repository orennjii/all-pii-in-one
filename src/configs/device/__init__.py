#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设备配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class DeviceConfig(BaseConfig):
    """设备配置"""
    use_gpu: bool = Field(default=True, description="是否优先使用GPU")
    gpu_id: int = Field(default=0, description="使用的GPU ID")
    use_fp16: bool = Field(default=True, description="是否启用半精度推理")
