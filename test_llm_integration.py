#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM识别器集成测试脚本

完整测试LLM识别器的各个组件：
1. 配置系统集成
2. 客户端工厂创建
3. 解析器工厂创建
4. 提示词加载
5. 实体识别流程
6. 与presidio的集成
"""

import sys
import os
from pathlib import Path
from typing import List

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import (
    LLMRecognizerConfig, LLMClientConfig, LLMParsersConfig, LLMPromptsConfig
)
from src.processors.text_processor.recognizers.llm.clients.client_factory import create_llm_client
from src.processors.text_processor.recognizers.llm.parsers.parser_factory import create_parser
from src.processors.text_processor.recognizers.llm.prompts.loader import PromptLoader
from src.processors.text_processor.recognizers.llm.recognizer import LLMRecognizer

logger = get_module_logger(__name__)


def test_configuration_system():
    """测试配置系统"""
    logger.info("=== 测试配置系统 ===")
    
    try:
        # 测试LLM客户端配置
        client_config = LLMClientConfig(
            type="gemini",
            model_name_or_path="gemini-2.5-flash",
            api_key="test-key",
            temperatures=0.7,
            max_tokens=1000
        )
        logger.info(f"✓ LLM客户端配置创建成功: {client_config.type}")
        
        # 测试解析器配置
        parser_config = LLMParsersConfig()
        logger.info(f"✓ 解析器配置创建成功: {parser_config.default_parser}")
        
        # 测试提示词配置
        prompts_config = LLMPromptsConfig()
        logger.info(f"✓ 提示词配置创建成功: {prompts_config.prompt_template_path}")
        
        # 测试LLM识别器配置
        llm_config = LLMRecognizerConfig(
            enabled=True,
            client=client_config,
            parsers=parser_config,
            prompts=prompts_config
        )
        logger.info(f"✓ LLM识别器配置创建成功: enabled={llm_config.enabled}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 配置系统测试失败: {str(e)}")
        return False


def test_client_factory():
    """测试客户端工厂"""
    logger.info("=== 测试客户端工厂 ===")
    
    try:
        # 创建测试配置
        config = LLMClientConfig(
            type="gemini",
            model_name_or_path="gemini-2.5-flash",
            api_key="test-key"
        )
        
        # 测试客户端创建（不实际初始化）
        client = create_llm_client(config)
        logger.info(f"✓ 客户端创建成功: {client.__class__.__name__}")
        
        # 测试客户端属性
        assert hasattr(client, 'generate'), "客户端缺少generate方法"
        assert hasattr(client, 'load'), "客户端缺少load方法"
        logger.info("✓ 客户端接口验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 客户端工厂测试失败: {str(e)}")
        return False


def test_parser_factory():
    """测试解析器工厂"""
    logger.info("=== 测试解析器工厂 ===")
    
    try:
        # 创建测试配置
        config = LLMParsersConfig()
        
        # 测试解析器创建
        parser = create_parser("gemini", config)
        logger.info(f"✓ 解析器创建成功: {parser.__class__.__name__}")
        
        # 测试解析器接口
        assert hasattr(parser, 'parse'), "解析器缺少parse方法"
        assert hasattr(parser, 'post_process_entities'), "解析器缺少post_process_entities方法"
        logger.info("✓ 解析器接口验证通过")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 解析器工厂测试失败: {str(e)}")
        return False


def test_prompt_loader():
    """测试提示词加载器"""
    logger.info("=== 测试提示词加载器 ===")
    
    try:
        # 创建提示词加载器
        template_path = project_root / "config" / "prompt_template.json"
        loader = PromptLoader(str(template_path))
        logger.info(f"✓ 提示词加载器创建成功，模板路径: {template_path}")
        
        # 测试模板加载
        template = loader.load_prompt_template("pii_detection")
        logger.info(f"✓ 提示词模板加载成功: {template.name}")
        
        # 测试变量验证
        required_vars = template.get_required_variables()
        logger.info(f"✓ 模板所需变量: {required_vars}")
        
        # 测试模板格式化
        if required_vars:
            test_vars = {var: f"test_{var}" for var in required_vars}
            formatted = template.format(**test_vars)
            logger.info(f"✓ 模板格式化成功，长度: {len(formatted)}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 提示词加载器测试失败: {str(e)}")
        return False


def test_llm_recognizer_creation():
    """测试LLM识别器创建"""
    logger.info("=== 测试LLM识别器创建 ===")
    
    try:
        # 创建完整配置
        client_config = LLMClientConfig(
            type="gemini",
            model_name_or_path="gemini-2.5-flash",
            api_key="test-key"
        )
        
        parser_config = LLMParsersConfig()
        prompts_config = LLMPromptsConfig()
        
        llm_config = LLMRecognizerConfig(
            enabled=True,
            client=client_config,
            parsers=parser_config,
            prompts=prompts_config
        )
        
        # 创建LLM识别器
        recognizer = LLMRecognizer(config=llm_config)
        logger.info(f"✓ LLM识别器创建成功: {recognizer.name}")
        
        # 测试识别器属性
        assert hasattr(recognizer, 'analyze'), "识别器缺少analyze方法"
        assert hasattr(recognizer, 'supported_entities'), "识别器缺少supported_entities属性"
        logger.info(f"✓ 支持的实体类型: {recognizer.supported_entities}")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ LLM识别器创建测试失败: {str(e)}")
        return False


def test_mock_recognition():
    """测试模拟识别流程（不调用真实API）"""
    logger.info("=== 测试模拟识别流程 ===")
    
    try:
        # 创建测试用的模拟响应
        mock_response = '''
        [
            {
                "entity_type": "PERSON",
                "value": "张三",
                "start": 0,
                "end": 2,
                "confidence": 0.95
            },
            {
                "entity_type": "PHONE_NUMBER", 
                "value": "13812345678",
                "start": 10,
                "end": 21,
                "confidence": 0.9
            }
        ]
        '''
        
        # 创建解析器并测试解析
        config = LLMParsersConfig()
        parser = create_parser("gemini", config)
        
        # 解析模拟响应
        result = parser.parse(mock_response, "张三的电话号码是13812345678")
        logger.info(f"✓ 解析模拟响应成功，找到 {len(result.entities)} 个实体")
        
        # 验证解析结果
        for entity in result.entities:
            logger.info(f"  - {entity.entity_type}: {entity.value} ({entity.start}-{entity.end})")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ 模拟识别流程测试失败: {str(e)}")
        return False


def test_presidio_integration():
    """测试与presidio的集成"""
    logger.info("=== 测试presidio集成 ===")
    
    try:
        from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
        
        # 创建LLM识别器
        client_config = LLMClientConfig(
            type="gemini",
            model_name_or_path="gemini-2.5-flash",
            api_key="test-key"
        )
        
        llm_config = LLMRecognizerConfig(
            enabled=True,
            client=client_config
        )
        
        recognizer = LLMRecognizer(config=llm_config)
        
        # 创建识别器注册表
        registry = RecognizerRegistry()
        registry.add_recognizer(recognizer)
        
        logger.info(f"✓ LLM识别器已添加到presidio注册表")
        logger.info(f"✓ 注册表中的识别器: {[r.name for r in registry.recognizers]}")
        
        # 创建分析引擎（不执行实际分析）
        analyzer = AnalyzerEngine(registry=registry)
        logger.info(f"✓ presidio分析引擎创建成功")
        
        return True
        
    except Exception as e:
        logger.error(f"✗ presidio集成测试失败: {str(e)}")
        return False


def run_all_tests():
    """运行所有测试"""
    print("开始LLM识别器集成测试")
    logger.info("开始LLM识别器集成测试")
    print("=" * 50)
    
    tests = [
        ("配置系统", test_configuration_system),
        ("客户端工厂", test_client_factory),
        ("解析器工厂", test_parser_factory),
        ("提示词加载器", test_prompt_loader),
        ("LLM识别器创建", test_llm_recognizer_creation),
        ("模拟识别流程", test_mock_recognition),
        ("presidio集成", test_presidio_integration),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        logger.debug(f"\n开始测试: {test_name}")
        try:
            if test_func():
                passed += 1
                logger.debug(f"✓ {test_name} 测试通过")
            else:
                logger.error(f"✗ {test_name} 测试失败")
        except Exception as e:
            logger.error(f"✗ {test_name} 测试异常: {str(e)}")
    
    print("\n" + "=" * 50)
    print(f"测试完成: {passed}/{total} 通过")
    logger.info("\n" + "=" * 50)
    logger.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        print("🎉 所有测试都通过了！LLM识别器集成正常。")
        logger.info("🎉 所有测试都通过了！LLM识别器集成正常。")
        return True
    else:
        print(f"⚠️  有 {total - passed} 个测试失败，请检查相关组件。")
        logger.warning(f"⚠️  有 {total - passed} 个测试失败，请检查相关组件。")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
