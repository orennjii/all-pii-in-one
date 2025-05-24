#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM 提示词模块

提供用于不同LLM模型的提示词模板管理和加载功能。
"""

from src.processors.text_processor.recognizers.llm.prompts.loader import PromptLoader
from src.processors.text_processor.recognizers.llm.prompts.gemini_prompts import GeminiPromptManager

__all__ = [
    "PromptLoader",
    "GeminiPromptManager"
]