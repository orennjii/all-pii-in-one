#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理 Tab 模块

提供文本PII检测和匿名化的 Gradio Web 界面
"""

from .text_tab import create_text_tab, TextProcessorTab

__all__ = [
    "create_text_tab",
    "TextProcessorTab"
]