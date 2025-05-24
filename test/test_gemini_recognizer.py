#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini LLM识别器测试模块

该模块演示如何从app_config.yaml加载配置并使用GeminiEntityRecognizer类识别文本中的个人隐私信息
"""

import os
import sys
import logging
from typing import List, Dict, Any, Optional

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.configs import AppConfig
from src.processors.text_processor.recognizers.llm.gemini_recognizer import GeminiEntityRecognizer
from src.processors.text_processor.recognizers.llm.clients import GeminiClient
from src.processors.text_processor.recognizers.llm.parsers import GeminiParser
from src.commons.loggers import get_module_logger

logger = get_module_logger(__name__)

def test_gemini_recognizer():
    """
    测试Gemini识别器的主函数
    
    从app_config.yaml加载配置，并使用GeminiEntityRecognizer类识别文本中的个人隐私信息
    """
    # 从app_config.yaml加载配置
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                              "config", "app_config.yaml")
    app_config = AppConfig.load_from_yaml(config_path)
    
    # 获取LLM配置
    llm_config = app_config.text_processor.recognizers.llm
    
    # 如果LLM识别器未启用，则退出
    if not llm_config.enabled:
        logger.warning("LLM识别器未启用，请在app_config.yaml中启用")
        return
    
    # 创建Gemini客户端
    gemini_client = GeminiClient(
        model_name_or_path=llm_config.client.model_name_or_path,
        temperature=llm_config.client.temperature,
        max_tokens=llm_config.client.max_tokens,
        top_p=llm_config.client.top_p,
        top_k=llm_config.client.top_k,
    )
    
    # 创建Gemini解析器
    gemini_parser = GeminiParser(
        expected_format='json',
        fallback_to_text=True,
        default_score=0.8,
    )
    
    # 支持的实体类型
    supported_entities = [
        "PERSON", "ID_CARD", "PHONE_NUMBER", "EMAIL_ADDRESS", 
        "CREDIT_CARD", "BANK_ACCOUNT", "ADDRESS", "LOCATION"
    ]
    
    # 创建Gemini识别器
    gemini_recognizer = GeminiEntityRecognizer(
        config=llm_config,
        supported_entities=supported_entities,
    )
    
    # 手动设置客户端和解析器
    gemini_recognizer.llm_client = gemini_client
    gemini_recognizer.response_parser = gemini_parser
    
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
    registry.add_recognizer(gemini_recognizer)
    
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
    test_gemini_recognizer()
