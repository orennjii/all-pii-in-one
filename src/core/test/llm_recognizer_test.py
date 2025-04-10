import unittest
import json
import os
import sys
from unittest import mock
from typing import Dict, Any, List, Optional

# 导入项目模块
from src.core.llm_recognizer.llm_recognizer import LLMRecognizer
from src.core.llm_recognizer.llm_clients.gemini_client import GeminiClient
from src.core.llm_recognizer.parsers.gemini_parser import GeminiParser

# 导入presidio依赖
from presidio_analyzer import AnalyzerEngine, RecognizerResult, RecognizerRegistry

def test():
    gemini_client = GeminiClient(
        api_key="AIzaSyBRLhMDRxxpaZRI2cWE_03BCL1379I-FYE",
        model_name='models/gemini-2.0-flash-thinking-exp-01-21',
    )

    gemini_parser = GeminiParser(
        expected_format='text',
        fallback_to_text=True,
        default_score=0.8,
    )

    llm_recognizer = LLMRecognizer(
        supported_entities=["DATE_TIME", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"],
        llm_client=gemini_client,
        llm_parser=gemini_parser,
        prompt_template="请帮我找出以下文本中的所有命名实体,生成一个JSON对象,包含实体的类型和文本和打分. 例如: [{\"entity_type\": 'DATE_TIME', \"text\": \"2023年10月1日\", \"score\": 0.95}, {\"entity_type\": 'PERSON', \"text\": \"张三\", \"score\": 0.92}] 文本: {text}",
    )

    cn_text = """
    这是一份包含敏感信息的示例文本："""

    registry = RecognizerRegistry(supported_languages=["zh"])
    registry.add_recognizer(llm_recognizer)

    analyzer = AnalyzerEngine(registry=registry, supported_languages=["zh"])

    analyzed_results = analyzer.analyze(
        text=cn_text,
        language='zh',
        ad_hoc_recognizers=llm_recognizer,
    )

    print(analyzed_results)

if __name__ == "__main__":
    test()

