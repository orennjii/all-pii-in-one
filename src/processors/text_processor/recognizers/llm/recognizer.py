#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM实体识别器模块

该模块提供基于大语言模型的PII识别器实现，用于识别文本中的个人隐私信息。
"""

import json
from typing import List, Dict, Any, Optional, Tuple

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import LLMRecognizerConfig
from src.processors.text_processor.recognizers.llm.clients.base_client import BaseLLMClient
from src.processors.text_processor.recognizers.llm.clients.client_factory import create_llm_client
from src.processors.text_processor.recognizers.llm.prompts.loader import PromptLoader
from src.processors.text_processor.recognizers.llm.parsers.parser_factory import create_parser


logger = get_module_logger(__name__)


class LLMRecognizer(EntityRecognizer):
    """
    基于大语言模型的实体识别器
    
    使用LLM来识别文本中可能的PII实体，通过发送提示词到LLM并解析响应来完成识别。
    支持多种LLM后端，可通过配置灵活切换。
    """
    
    def __init__(
        self,
        config: Optional[LLMRecognizerConfig] = None,
        supported_entities: Optional[List[str]] = None,
        **kwargs
    ):
        """
        初始化LLM实体识别器
        
        Args:
            config: LLM识别器配置
            supported_entities: 支持识别的实体类型列表
            **kwargs: 其他参数
        """
        if not config:
            config = LLMRecognizerConfig()

        self.config = config
        
        # 默认支持的实体类型
        if not supported_entities:
            supported_entities = [
                "PERSON", "ID_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS", 
                "CREDIT_CARD", "BANK_ACCOUNT", "ADDRESS", "LOCATION"
            ]
                    
        # 初始化LLM客户端
        self.llm_client = create_llm_client(config.client)
        
        # 加载提示词模板
        self.prompt_loader = PromptLoader(config.prompts.prompt_template_path)
        
        # 初始化响应解析器
        self.response_parser = create_parser(config.client.type)
        
        # 初始化父类
        super().__init__(
            supported_entities=supported_entities,
            supported_language="zh",
            name="LLMEntityRecognizer"
        )
        
        logger.info(f"LLM实体识别器已初始化 (使用{self.llm_client.__class__.__name__})")
        
    def load(self) -> None:
        """加载识别器所需资源"""
        self.llm_client.load()
        logger.debug("LLM识别器资源已加载")
        
    def analyze(
        self,
        text: str,
        entities: Optional[List[str]] = None,
        nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        分析文本中的PII实体
        
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
            # 调用LLM
            llm_response = self.llm_client.generate(prompt)
            
            # 解析LLM响应
            llm_response_obj = self.response_parser.parse(llm_response, text)
            
            # 转换为RecognizerResult
            results = self.convert_to_recognizer_results(llm_response_obj.entities)
            
            logger.debug(f"LLM识别结果: 找到{len(results)}个实体")
            return results
            
        except Exception as e:
            logger.error(f"LLM识别过程出错: {str(e)}")
            return []
    
    def prepare_prompt(self, text: str, entities: List[str]) -> str:
        """
        准备发送给LLM的提示词
        
        Args:
            text: 待分析的文本
            entities: 要识别的实体类型列表
            
        Returns:
            str: 格式化后的提示词
        """
        # 获取基础模板
        prompt_template = self.prompt_loader.load_prompt_template("pii_detection")
        
        # 准备提示词变量
        prompt_vars = {
            "text": text,
            "entities": ", ".join(entities)
        }
        
        # 格式化提示词
        prompt = prompt_template.format(**prompt_vars)
        return prompt
    
    def convert_to_recognizer_results(
        self, 
        entity_matches: List
    ) -> List[RecognizerResult]:
        """
        将LLM解析结果转换为标准的RecognizerResult格式
        
        Args:
            entity_matches: EntityMatch对象列表
            
        Returns:
            List[RecognizerResult]: 标准格式的识别结果
        """
        results = []
        
        for entity_match in entity_matches:
            try:
                if hasattr(entity_match, 'entity_type'):
                    # EntityMatch对象
                    result = RecognizerResult(
                        entity_type=entity_match.entity_type,
                        start=entity_match.start,
                        end=entity_match.end,
                        score=entity_match.confidence
                    )
                    results.append(result)
                else:
                    # 字典格式（向后兼容）
                    entity_type = entity_match.get("entity_type")
                    start = entity_match.get("start")
                    end = entity_match.get("end")
                    score = entity_match.get("confidence", 0.8)
                    
                    if entity_type and isinstance(start, int) and isinstance(end, int):
                        result = RecognizerResult(
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            score=score
                        )
                        results.append(result)
            except Exception as e:
                logger.warning(f"转换识别结果时出错: {str(e)}")
                
        return results
