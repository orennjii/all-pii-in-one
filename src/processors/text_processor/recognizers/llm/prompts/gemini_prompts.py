#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini特定提示词模块

提供专为Google Gemini模型优化的提示词管理和加载功能。
"""

import os
from typing import Dict, Any, Optional, List, Union

from src.commons.loggers import get_module_logger
from src.processors.text_processor.recognizers.llm.prompts.loader import PromptLoader

logger = get_module_logger(__name__)

# Gemini模型提示词模板路径
GEMINI_PROMPT_TEMPLATE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    "templates", 
    "gemini_prompts.json"
)


class GeminiPromptManager:
    """
    Gemini提示词管理器
    
    专门为Google Gemini模型提供优化的提示词管理功能。
    包括模板加载、多语言支持、提示词优化等。
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        初始化Gemini提示词管理器
        
        Args:
            template_path: 模板文件路径，如果为None则使用默认路径
        """
        self.template_path = template_path or GEMINI_PROMPT_TEMPLATE_PATH
        self.prompt_loader = PromptLoader(self.template_path)
        self.language = "zh"  # 默认语言
        
        logger.debug(f"Gemini提示词管理器已初始化，使用模板: {self.template_path}")
        
    def set_language(self, language: str) -> None:
        """
        设置语言
        
        Args:
            language: 语言代码 (如 'zh', 'en')
        """
        supported_languages = ["zh", "en"]
        if language not in supported_languages:
            logger.warning(f"不支持的语言: {language}，将使用默认语言: zh")
            language = "zh"
            
        self.language = language
        logger.debug(f"已设置语言为: {self.language}")
        
    def get_pii_detection_prompt(
        self, 
        text: str, 
        entities: Union[List[str], str],
        context_type: str = "standard"
    ) -> str:
        """
        获取PII检测提示词
        
        Args:
            text: 待处理文本
            entities: 要检测的实体类型，可以是列表或逗号分隔的字符串
            context_type: 上下文类型，可选值: 'standard', 'structured', 
                          'context', 'document', 'complex'
                          
        Returns:
            str: 格式化后的提示词
        """
        # 确保entities是字符串类型
        if isinstance(entities, list):
            entities = ", ".join(entities)
            
        # 根据上下文类型和语言选择模板
        template_name = "pii_detection"  # 默认模板
        
        # 检查是否有特定语言版本的模板
        lang_template = f"pii_detection_{self.language}"
        if lang_template in self.prompt_loader.templates:
            template_name = lang_template
            
        # 检查是否有特定上下文类型的模板
        if context_type != "standard":
            context_template_map = {
                "structured": "pii_extraction_structured",
                "context": "pii_detection_with_context",
                "document": "document_pii_detection",
                "complex": "complex_pii_detection"
            }
            
            if context_type in context_template_map:
                specific_template = context_template_map[context_type]
                if specific_template in self.prompt_loader.templates:
                    template_name = specific_template
                    
        # 渲染模板
        return self.prompt_loader.render_template(
            template_name,
            text=text,
            entities=entities
        )
        
    def get_system_prompt(self) -> str:
        """
        获取系统提示词
        
        Returns:
            str: 系统提示词
        """
        return self.prompt_loader.load_prompt_template("system_prompt")
    
    def optimize_prompt_for_gemini(self, prompt: str) -> str:
        """
        针对Gemini模型优化提示词
        
        根据Gemini模型的特点，对提示词进行优化调整。
        
        Args:
            prompt: 原始提示词
            
        Returns:
            str: 优化后的提示词
        """
        # 这里可以实现针对Gemini的提示词优化逻辑
        # 例如，添加特定的格式、调整指令的清晰度等
        
        # 当前仅做简单优化，确保提示词结尾提醒仅返回JSON
        if "JSON" in prompt and not prompt.strip().endswith("JSON"):
            prompt += "\n\n请记住，只返回JSON格式的结果，不需要任何额外的解释或注释。"
            
        return prompt
    
    def create_pii_extraction_prompt(
        self, 
        text: str, 
        entities: Optional[List[str]] = None,
        language: Optional[str] = None
    ) -> str:
        """
        创建PII提取提示词
        
        Args:
            text: 待处理文本
            entities: 要提取的实体类型列表（可选）
            language: 语言（可选，覆盖默认语言）
            
        Returns:
            str: 格式化后的提示词
        """
        # 临时设置语言（如果指定）
        original_language = self.language
        if language:
            self.set_language(language)
            
        # 准备实体类型列表
        if entities:
            entities_str = ", ".join(entities)
        else:
            default_entities = [
                "PERSON", "ID_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS", 
                "CREDIT_CARD", "BANK_ACCOUNT", "ADDRESS", "LOCATION"
            ]
            entities_str = ", ".join(default_entities)
            
        # 渲染模板
        prompt = self.prompt_loader.render_template(
            "pii_extraction",
            text=text,
            entities=entities_str
        )
        
        # 恢复原始语言设置
        if language:
            self.set_language(original_language)
            
        # 优化提示词
        return self.optimize_prompt_for_gemini(prompt)
    
    def create_custom_prompt(
        self, 
        template_name: str, 
        **kwargs
    ) -> str:
        """
        创建自定义提示词
        
        使用指定的模板和参数创建自定义提示词。
        
        Args:
            template_name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            str: 格式化后的提示词
            
        Raises:
            KeyError: 如果指定的模板不存在
        """
        try:
            prompt = self.prompt_loader.render_template(template_name, **kwargs)
            return self.optimize_prompt_for_gemini(prompt)
        except KeyError as e:
            logger.error(f"模板 '{template_name}' 不存在: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"创建自定义提示词失败: {str(e)}")
            raise