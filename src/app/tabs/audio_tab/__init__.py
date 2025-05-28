#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理 Gradio UI 模块

提供音频PII检测与匿名化的Web界面
"""

from .audio_tab import AudioProcessorTab, create_audio_tab

__all__ = [
    'AudioProcessorTab',
    'create_audio_tab'
]