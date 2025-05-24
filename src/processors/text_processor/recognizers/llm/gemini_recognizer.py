#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini实体识别器模块

该模块提供基于Google Gemini大语言模型的PII识别器实现，
使用专为Gemini优化的提示词和解析策略。
"""

from typing import List, Dict, Any, Optional

from presidio_analyzer import RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers import LLMRecognizerConfig
from src.processors.text_processor.recognizers.llm.recognizer import LLMRecognizer
from src.processors.text_processor.recognizers.llm.prompts.gemini_prompts import GeminiPromptManager
from src.processors.text_processor.recognizers.llm.parsers.gemini_parser import GeminiResponseParser

logger = get_module_logger(__name__)


class GeminiEntityRecognizer(LLMRecognizer):
    """
    基于Google Gemini模型的实体识别器
    
    继承自基础LLM识别器，但使用专为Gemini优化的提示词和解析策略。
    提供更高的识别准确率和更好的多语言支持。
    """
    
    def __init__(
        self,
        config: Optional[LLMRecognizerConfig] = None,
        supported_entities: Optional[List[str]] = None,
        **kwargs
    ):
        """
        初始化Gemini实体识别器
        
        Args:
            config: LLM识别器配置
            supported_entities: 支持识别的实体类型列表
            **kwargs: 其他参数
        """
        # 调用父类初始化方法
        super().__init__(config, supported_entities, **kwargs)
        
        # 替换为Gemini特定的提示词管理器和解析器
        self.prompt_manager = GeminiPromptManager()
        self.response_parser = GeminiResponseParser()
        
        # 设置语言（从配置中读取或使用默认值）
        language = getattr(config, "language", "zh") if config else "zh"
        self.prompt_manager.set_language(language)
        
        logger.info(f"Gemini实体识别器已初始化，使用语言: {language}")
    
    def prepare_prompt(self, text: str, entities: List[str]) -> str:
        """
        准备发送给LLM的提示词，使用Gemini优化的提示词
        
        Args:
            text: 待分析的文本
            entities: 要识别的实体类型列表
            
        Returns:
            str: 格式化后的提示词
        """
        # 确定上下文类型
        context_type = "standard"
        if len(text) > 1000:
            context_type = "document"  # 长文本使用文档模式
        elif any(c in text for c in ['|', '+', '表', '列表']):
            context_type = "structured"  # 可能包含表格的文本使用结构化模式
            
        # 使用Gemini提示词管理器生成优化的提示词
        prompt = self.prompt_manager.get_pii_detection_prompt(
            text=text,
            entities=entities,
            context_type=context_type
        )
        
        logger.debug(f"已为Gemini准备提示词，使用{context_type}模式")
        return prompt
        
    def analyze(
        self,
        text: str,
        entities: List[str] = None,
        nlp_artifacts: NlpArtifacts = None
    ) -> List[RecognizerResult]:
        """
        分析文本中的PII实体，使用Gemini优化的策略
        
        Args:
            text: 待分析的文本
            entities: 要识别的实体类型列表，如果为None则使用所有支持的实体
            nlp_artifacts: NLP分析结果
            
        Returns:
            List[RecognizerResult]: 识别结果列表
        """
        if not text:
            return []
            
        if not entities:
            entities = self.supported_entities
        else:
            # 过滤出支持的实体
            entities = [e for e in entities if e in self.supported_entities]
            
        if not entities:
            return []
            
        # 准备提示词
        prompt = self.prepare_prompt(text, entities)
        
        try:
            # 调用LLM，添加系统提示词
            system_prompt = self.prompt_manager.get_system_prompt()
            llm_response = self.llm_client.generate(
                prompt, 
                system_prompt=system_prompt
            )
            
            # 解析LLM响应
            recognition_results = self.response_parser.parse(llm_response, text, entities)
            
            # 转换为RecognizerResult
            results = self.convert_to_recognizer_results(recognition_results)
            
            logger.debug(f"Gemini识别结果: 找到{len(results)}个实体")
            return results
            
        except Exception as e:
            logger.error(f"Gemini识别过程出错: {str(e)}")
            return []