#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理 Gradio UI 模块

提供图像PII检测与编辑的Web界面
"""

from .image_tab import ImageProcessorTab, create_image_tab

__all__ = [
    'ImageProcessorTab',
    'create_image_tab'
]
