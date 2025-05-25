#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PII匿名化器模块

提供PII（个人隐私信息）匿名化的核心功能，支持多种匿名化策略。
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, EngineResult
from presidio_analyzer import RecognizerResult

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor import AnonymizerConfig

class PresidioAnonymizer:
    """基于Presidio的PII匿名化器 - 简化版本
    
    直接使用Presidio库进行PII匿名化，使用统一的匿名化策略。
    """
    
    def __init__(self, config: AnonymizerConfig):
        """
        初始化Presidio匿名化器
        
        Args:
            config: 匿名化器配置
        """
        self.config = config
        self.logger = get_module_logger(__name__)
        self._anonymizer = None
        self._setup_anonymizer()
    
    def _setup_anonymizer(self) -> None:
        """设置Presidio匿名化器"""
        try:
            # 创建匿名化器引擎
            self._anonymizer = AnonymizerEngine()
            
            self.logger.info("Presidio anonymizer initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Presidio anonymizer: {e}")
            raise
    
    def anonymize(
        self,
        text: str,
        analyzer_results: List[RecognizerResult],
        operators: Optional[Dict[str, str]] = None
    ) -> EngineResult:
        """
        使用Presidio匿名化文本中的PII
        
        Args:
            text: 原始文本
            analyzer_results: Presidio分析器检测到的PII结果
            operators: 实体类型到操作符的映射（可选）
            
        Returns:
            匿名化结果
        """
        if not self._anonymizer:
            raise RuntimeError("Anonymizer not initialized")
        
        try:
            # 创建操作符配置
            operator_configs = self._create_operator_config(operators)
            
            # 执行匿名化
            anonymize_result = self._anonymizer.anonymize(
                text=text,
                analyzer_results=analyzer_results,
                operators=operator_configs
            )

            self.logger.debug(f"Anonymized {len(anonymize_result.items)} PII entities")
            return anonymize_result

        except Exception as e:
            self.logger.error(f"Error during anonymization: {e}")
            # 返回原始文本作为fallback
            return EngineResult(text=text, items=[])

    def _create_operator_config(
        self, 
        operators: Optional[Dict[str, str]] = None
    ) -> Dict[str, OperatorConfig]:
        """
        创建操作符配置 - 简化版本
        
        Args:
            operators: 实体类型到操作符的映射
            
        Returns:
            操作符配置字典
        """
        # 使用统一的匿名化策略
        strategy = self.config.anonymization_strategy
        params = self.config.operator_params.copy()
        
        # 如果指定了特定操作符，则使用指定的
        if operators:
            operator_configs = {}
            for entity_type, operator_name in operators.items():
                operator_configs[entity_type] = OperatorConfig(
                    operator_name=operator_name,
                    params=self._get_params_for_operator(operator_name)
                )
            return operator_configs
        else:
            # 使用默认策略创建通用配置
            default_config = OperatorConfig(
                operator_name=strategy,
                params=self._get_params_for_operator(strategy)
            )
            
            # 为所有可能的实体类型使用相同的配置
            return {"DEFAULT": default_config}
    
    def _get_params_for_operator(self, operator_name: str) -> Dict[str, Any]:
        """
        根据操作符名称获取参数
        
        Args:
            operator_name: 操作符名称
            
        Returns:
            操作符参数
        """
        params = self.config.operator_params
        
        if operator_name == "replace":
            return {"new_value": params.get("new_value", "[REDACTED]")}
        elif operator_name == "mask":
            return {
                "masking_char": params.get("masking_char", "*"),
                "chars_to_mask": params.get("chars_to_mask", -1),
                "from_end": params.get("from_end", True)
            }
        elif operator_name == "hash":
            return {"hash_type": params.get("hash_type", "sha256")}
        elif operator_name == "redact":
            return {}
        else:
            return {}
    
    def batch_anonymize(
        self,
        texts: List[str],
        analyzer_results_list: List[List[RecognizerResult]],
        operators: Optional[Dict[str, str]] = None
    ) -> List[EngineResult]:
        """
        批量匿名化文本
        
        Args:
            texts: 原始文本列表
            analyzer_results_list: 每个文本对应的分析结果列表
            operators: 实体类型到操作符的映射
            
        Returns:
            匿名化结果列表
        """
        results = []
        for text, analyzer_results in zip(texts, analyzer_results_list):
            try:
                result = self.anonymize(text, analyzer_results, operators)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error anonymizing text: {e}")
                # 返回原始文本作为fallback
                results.append(EngineResult(text=text, items=[]))
        return results