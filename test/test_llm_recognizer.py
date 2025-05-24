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
    
    # 获取LLM配置
    llm_config = app_config.text_processor.recognizers.llm
    
    # 如果LLM识别器未启用，则退出
    if not llm_config.enabled:
        logger.warning("LLM识别器未启用，请在app_config.yaml中启用")
        return
    
    # 创建LLM客户端，显式指定类型为小写的"gemini"
    client_config = LLMClientConfig(
        type="gemini",
        api_key=llm_config.client.api_key,
        model=llm_config.client.model,
        timeout=llm_config.client.timeout,
        max_retries=llm_config.client.max_retries
    )
    llm_client = create_llm_client(client_config)
    
    # 创建解析器配置
    from src.configs.processors.text_processor.recognizers.llm.parsers_config import LLMParsersConfig
    parser_config = LLMParsersConfig(
        parser_type="gemini",
        json_strict_mode=False,
        min_confidence=0.8,
        gemini_response_formats=["json", "text"]
    )
    
    # 创建LLM解析器
    parser = GeminiParser(config=parser_config)
    
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
        llm_client=llm_client,
        parser=parser
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
    test_text = """
    张三的手机号是13912345678，他的邮箱是zhangsan@example.com。
    他住在北京市海淀区中关村南大街5号，身份证号码是110101199001011234。
    他的信用卡号是6222020111122223333，银行账号是6225881234567890。
    """
    
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
    
    return results

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_llm_recognizer()