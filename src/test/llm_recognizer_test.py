import unittest
import json
import os
import logging
import sys
from unittest import mock
from typing import Dict, Any, List, Optional

try:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:17072"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:17072"
    logging.info("已成功设置代理")
except Exception as e:
    logging.warning(f"Error setting environment variables: {e}")

from src.core.llm_recognizer.llm_recognizer import LLMRecognizer
from src.core.llm_recognizer.llm_clients.gemini_client import GeminiClient
from src.core.llm_recognizer.parsers.gemini_parser import GeminiParser
from src.core.llm_recognizer.config import BaseLLMSettings

# 导入presidio依赖
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine


def test():
    text = """
    这是一份包含敏感信息的示例文本：

    我的姓名是张三，来自北京。
    我的车牌号是京A12345，手机号码是13812345678。
    我的身份证号是210102199203183096，银行卡号是4929717705917895。

    2023年10月15日，我访问了www.example.com并发送邮件到test@example.com，IP地址是192.168.1.1。
    """
    gemini_config = BaseLLMSettings(
        model_name='models/gemini-2.0-flash-thinking-exp-01-21',
        supported_entities=["DATE_TIME", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"],
    )

    gemini_client = GeminiClient(
        api_key="AIzaSyBRLhMDRxxpaZRI2cWE_03BCL1379I-FYE",
        model_name='models/gemini-2.0-flash-thinking-exp-01-21',
    )

    gemini_parser = GeminiParser(
        expected_format='json',
        fallback_to_text=True,
        default_score=0.8,
    )

    llm_recognizer = LLMRecognizer(
        supported_entities=["DATE_TIME", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS", "ID_CARD", "BANK_CARD"],
        llm_client=gemini_client,
        parser=gemini_parser,
        prompt_template=gemini_config.prompt_template,
    )

    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{
            "lang_code": "zh",
            "model_name": "zh_core_web_lg"
        }]
    }).create_engine()

    registry = RecognizerRegistry(supported_languages=["zh"])
    registry.add_recognizer(llm_recognizer)

    analyzer = AnalyzerEngine(
        registry=registry,
        supported_languages=["zh"],
        nlp_engine=nlp_engine
    )

    analyzed_results = analyzer.analyze(
        entities=gemini_config.supported_entities,
        text=text,
        language='zh',
    )

    print(analyzed_results)


if __name__ == "__main__":
    test()

