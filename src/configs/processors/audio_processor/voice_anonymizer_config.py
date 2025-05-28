#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音匿名化配置模块 - 简化版

提供语音匿名化处理的基础配置定义。
"""

from pydantic import Field

from src.configs.base_config import BaseConfig


class VoiceAnonymizerConfig(BaseConfig):
    """语音匿名化配置 - 简化版"""
    enabled: bool = Field(
        default=True,
        description="是否启用语音匿名化功能"
    )
    sample_rate: int = Field(
        default=22050,
        description="音频采样率（Hz）"
    )
    beep_frequency: float = Field(
        default=1000.0,
        description="蜂鸣音频率（Hz）",
        gt=0.0
    )
    beep_amplitude: float = Field(
        default=0.3,
        description="蜂鸣音音量（0-1）",
        ge=0.0,
        le=1.0
    )
    fade_duration: float = Field(
        default=0.05,
        description="淡入淡出时长（秒）",
        ge=0.0
    )
