#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理器配置模块（兼容层）
为保持向后兼容，现在从新的模块结构中导入配置类
"""

# 导入必要的类型和字段
from typing import Optional
from pydantic import Field

# 导入基础配置类
from src.configs.base_config import BaseConfig

# 从新的文件结构导入配置类
from src.configs.processors.text_processor import TextProcessorConfig
from src.configs.processors.text_processor.recognizers import RecognizersConfig
from src.configs.processors.text_processor.recognizers.pattern.pattern import PatternRecognizerConfig
from src.configs.processors.text_processor.recognizers.llm import LLMRecognizerConfig
from src.configs.processors.text_processor.recognizers.llm.client import LLMClientConfig
from src.configs.processors.text_processor.recognizers.llm.prompts import LLMPromptsConfig
from src.configs.processors.text_processor.recognizers.llm.parsers import LLMParsersConfig


class PatternRecognizerConfig(BaseConfig):
    """模式识别器配置"""
    enabled: bool = Field(default=True, description="是否启用模式识别器")
    id_card: bool = Field(default=True, description="是否识别身份证号")
    phone_number: bool = Field(default=True, description="是否识别电话号码")
    bank_card: bool = Field(default=True, description="是否识别银行卡号")
    car_plate: bool = Field(default=True, description="是否识别车牌号")
    url: bool = Field(default=True, description="是否识别URL")


class LLMClientConfig(BaseConfig):
    """LLM客户端配置"""
    type: str = Field(default="HuggingFace", description="LLM客户端类型")
    model_name_or_path: str = Field(default="meta-llama/Llama-2-7b-chat-hf", description="模型名称或路径")
    temperatures: float = Field(default=0.7, description="LLM温度参数")
    api_key: Optional[str] = Field(default=None, description="API密钥")
    endpoint: Optional[str] = Field(default=None, description="API端点")


class LLMPromptsConfig(BaseConfig):
    """LLM提示词配置"""
    prompt_template_path: str = Field(
        default="src/processors/text_processor/recognizers/llm/prompts/default_prompt.json",
        description="提示词模板路径"
    )


class LLMParsersConfig(BaseConfig):
    """LLM解析器配置"""
    example: str = Field(default="default", description="解析器示例配置")


class LLMRecognizerConfig(BaseConfig):
    """LLM识别器配置"""
    enabled: bool = Field(default=False, description="是否启用LLM识别器")
    client: LLMClientConfig = Field(default_factory=LLMClientConfig, description="LLM客户端配置")
    prompts: LLMPromptsConfig = Field(default_factory=LLMPromptsConfig, description="LLM提示词配置")
    parsers: LLMParsersConfig = Field(default_factory=LLMParsersConfig, description="LLM解析器配置")


class RecognizersConfig(BaseConfig):
    """识别器配置"""
    pattern: PatternRecognizerConfig = Field(default_factory=PatternRecognizerConfig, description="模式识别器配置")
    llm: LLMRecognizerConfig = Field(default_factory=LLMRecognizerConfig, description="LLM识别器配置")


class TextProcessorConfig(BaseConfig):
    """文本处理器配置"""
    recognizers: RecognizersConfig = Field(default_factory=RecognizersConfig, description="识别器配置")
    # 可以在这里添加其他文本处理器相关的配置
