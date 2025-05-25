#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM识别器测试模块

该模块演示如何从app_config.yaml加载配置并使用LLMRecognizer类识别文本中的个人隐私信息
可以通过以下方式运行:
python test/test_llm_recognizer.py
或
PYTHONPATH=. python -m test.test_llm_recognizer
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine

from src.commons.utils import find_project_root
from src.configs import AppConfig
from src.processors.text_processor.recognizers.llm import LLMRecognizer
from src.processors.text_processor.recognizers.llm.clients import create_llm_client
from src.processors.text_processor.recognizers.llm.parsers import GeminiParser
from src.configs.processors.text_processor.recognizers.llm.client_config import LLMClientConfig
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)

current_path = Path(__file__).resolve()
project_root = find_project_root(current_path)

def test_llm_recognizer():
    """
    测试LLM识别器的主函数
    
    从app_config.yaml加载配置，并使用LLMRecognizer类识别文本中的个人隐私信息
    """
    
    # 从app_config.yaml加载配置
    config_path = project_root / "config" / "app_config.yaml"
    app_config = AppConfig.load_from_yaml(str(config_path))
    print(f"配置路径: {config_path}")
    print(f"加载配置: {app_config}")
    
    # 获取LLM配置
    llm_config = app_config.text_processor.recognizers.llm_recognizer
    
    # 如果LLM识别器未启用，则退出
    if not llm_config.enabled:
        logger.warning("LLM识别器未启用，请在app_config.yaml中启用")
        return
    
    # 支持的实体类型
    supported_entities = [
        "PERSON", "ID_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS", 
        "CREDIT_CARD", "BANK_ACCOUNT", "ADDRESS", "LOCATION"
    ]
    
    # 创建LLM识别器
    # 直接使用自定义解析器而不是通过工厂创建
    llm_recognizer = LLMRecognizer(
        config=llm_config,
        supported_entities=supported_entities,
    )
    
    # 创建NLP引擎
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{
            "lang_code": "zh",
            "model_name": "zh_core_web_lg"
        }]
    }).create_engine()
    
    # 创建识别器注册表
    registry = RecognizerRegistry(supported_languages=["zh"])
    registry.add_recognizer(llm_recognizer)
    
    # 创建分析引擎
    analyzer = AnalyzerEngine(
        registry=registry,
        supported_languages=["zh"],
        nlp_engine=nlp_engine
    )
    
    # 测试文本
    test_text = "张三的手机号是13912345678,他的邮箱是zhangsan@example.com。他住在北京市海淀区中关村南大街5号，身份证号码是110101199001011234。他的信用卡号是6222020111122223333，银行账号是6225881234567890。"
    
    # 分析文本
    logger.info("开始分析文本中的个人隐私信息...")
    results = analyzer.analyze(
        text=test_text,
        entities=supported_entities,
        language='zh',
    )
    
    # 打印结果
    if results:
        logger.info(f"找到 {len(results)} 个个人隐私信息:")
        for result in results:
            entity_value = test_text[result.start:result.end]
            logger.info(f"类型: {result.entity_type}, 值: {entity_value}, " 
                      f"位置: {result.start}-{result.end}, 置信度: {result.score}")
    else:
        logger.info("未找到任何个人隐私信息")
    
    anonymizer_engine = AnonymizerEngine()

    anonymized_text = anonymizer_engine.anonymize(
        text=test_text,
        analyzer_results=results,
    )

    logger.info(f"文本匿名化结果: {anonymized_text}")

if __name__ == "__main__":
    test_llm_recognizer()