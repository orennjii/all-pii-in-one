#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PII分析器模块

提供PII（个人隐私信息）检测的核心功能，支持多种检测策略和识别器。
"""

from typing import List, Dict, Any, Optional, Union
from functools import lru_cache

from presidio_analyzer import AnalyzerEngine, RecognizerResult, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider, NlpEngine

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor import (
    TextProcessorConfig,
    AnalyzerConfig
)
from src.processors.text_processor.recognizers.pattern import (
    BankCardRecognizer,
    CarPlateNumberRecognizer,
    IDCardRecognizer,
    PhoneNumberRecognizer,
    URLRecognizer
)
from src.processors.text_processor.recognizers.llm import LLMRecognizer

logger = get_module_logger(__name__)

class PresidioAnalyzer():
    """基于Presidio的PII分析器
    
    使用微软的Presidio库进行PII检测，支持多种语言和自定义识别器。
    """
    
    def __init__(self, config: TextProcessorConfig):
        """
        初始化Presidio分析器
        
        Args:
            config: 分析器配置
        """
        self.config = config
        self._analyzer: AnalyzerEngine
        self._nlp_engine: NlpEngine
        self._registry: RecognizerRegistry

        self._setup_analyzer()
    
    def _setup_analyzer(self) -> None:
        """设置Presidio分析器"""
        try:
            # 设置NLP引擎
            self._setup_nlp_engine()
            self._setup_registry()
            
            # 创建分析器引擎
            self._analyzer = AnalyzerEngine(
                registry=self._registry,
                nlp_engine=self._nlp_engine,
                supported_languages=self.config.analyzer.supported_languages
            )
            
            logger.info("Presidio analyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Presidio analyzer: {e}")
            raise
    
    def _setup_nlp_engine(self) -> None:
        """设置NLP引擎"""
        try:
            # 配置NLP引擎提供者
            nlp_configuration = {
                "nlp_engine_name": "spacy",
                "models": [
                    {"lang_code": "zh", "model_name": "zh_core_web_sm"},
                ]
            }
            
            provider = NlpEngineProvider(nlp_configuration=nlp_configuration)
            self._nlp_engine = provider.create_engine()
            
        except Exception as e:
            logger.warning(f"Failed to setup NLP engine: {e}, using default")
            # 使用默认配置
            provider = NlpEngineProvider()
            self._nlp_engine = provider.create_engine()
    
    def _setup_registry(self) -> None:
        """设置识别器注册表"""
        try:
            # 创建识别器注册表
            self._registry = RecognizerRegistry(
                supported_languages=self.config.analyzer.supported_languages
            )
            
            # 添加内置识别器
            self._register_custom_recognizers()
            
        except Exception as e:
            logger.error(f"Failed to setup recognizer registry: {e}")
            raise
    
    def _register_custom_recognizers(self) -> None:
        """注册自定义识别器"""
        try:
            # 这里可以注册pattern recognizers和LLM recognizers
            # 从recognizers模块导入并注册
            if self.config.analyzer.enable_pattern_recognizers:
                self._register_pattern_recognizers()
            
            if self.config.analyzer.enable_llm_recognizers:
                self._register_llm_recognizers()
                
        except Exception as e:
            logger.error(f"Failed to register custom recognizers: {e}")
    
    def _register_pattern_recognizers(self) -> None:
        """注册模式识别器"""
        try:
            # 注册各种模式识别器
            recognizers = [
                # BankCardRecognizer(),
                # CarPlateNumberRecognizer(),
                # IDCardRecognizer(),
                # PhoneNumberRecognizer(),
                # URLRecognizer()
            ]
            
            for recognizer in recognizers:
                self._registry.add_recognizer(recognizer)
            
            logger.info(f"Registered {len(recognizers)} pattern recognizers")
            
        except Exception as e:
            logger.error(f"Failed to register pattern recognizers: {e}")
    
    def _register_llm_recognizers(self) -> None:
        """注册LLM识别器"""
        try:
            llm_recognizer = LLMRecognizer(
                config=self.config.recognizers.llm_recognizer,
                entitites=self.config.supported_entities,
            )
            self._registry.add_recognizer(llm_recognizer)
            
        except Exception as e:
            logger.error(f"Failed to register LLM recognizers: {e}")
    
    def analyze(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RecognizerResult]:
        """
        使用Presidio分析文本中的PII
        
        Args:
            text: 待分析的文本
            language: 语言代码
            entities: 要检测的实体类型列表
            score_threshold: 置信度阈值
            
        Returns:
            检测到的PII列表
        """
        if not self._analyzer:
            raise RuntimeError("Analyzer not initialized")
        
        # 设置默认参数
        if language is None:
            language = self.config.analyzer.default_language
        
        if score_threshold is None:
            score_threshold = self.config.analyzer.default_score_threshold

        # 应用实体过滤
        if entities is None:
            entities = self.config.supported_entities

        try:
            # 执行分析
            results = self._analyzer.analyze(
                text=text,
                language=language,
                entities=entities,
                score_threshold=score_threshold
            )
            
            # 过滤被拒绝的实体类型
            filtered_results = []
            for result in results:
                if (self.config.analyzer.denied_entities and 
                    result.entity_type in self.config.analyzer.denied_entities):
                    continue
                filtered_results.append(result)
            
            logger.debug(f"Found {len(filtered_results)} PII entities in text")
            return filtered_results
            
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            return []
    
    def batch_analyze(
        self,
        texts: List[str],
        language: Optional[str] = None,
        entities: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[List[RecognizerResult]]:
        """
        批量分析文本
        
        Args:
            texts: 待分析的文本列表
            language: 语言代码
            entities: 要检测的实体类型列表
            score_threshold: 置信度阈值
            
        Returns:
            每个文本的检测结果列表
        """
        results = []
        for text in texts:
            try:
                result = self.analyze(text, language, entities, score_threshold)
                results.append(result)
            except Exception as e:
                logger.error(f"Error analyzing text: {e}")
                results.append([])
        return results

    @lru_cache(maxsize=None)
    def get_supported_entities(self) -> List[str]:
        """
        获取支持的实体类型列表
        
        Returns:
            支持的实体类型列表
        """
        if not self._analyzer:
            return []
        
        try:
            return self._analyzer.get_supported_entities()
        except Exception as e:
            logger.error(f"Error getting supported entities: {e}")
            return []