#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
模式识别器配置模块
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class PatternRecognizerConfig(BaseConfig):
    """模式识别器配置"""
    enabled: bool = Field(default=True, description="是否启用模式识别器")
    id_card: bool = Field(default=True, description="是否识别身份证号")
    phone_number: bool = Field(default=True, description="是否识别电话号码")
    bank_card: bool = Field(default=True, description="是否识别银行卡号")
    car_plate: bool = Field(default=True, description="是否识别车牌号")
    url: bool = Field(default=True, description="是否识别URL")
