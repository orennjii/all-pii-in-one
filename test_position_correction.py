#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
测试位置修正功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.processors.text_processor.recognizers.llm.parsers.entity_match import EntityMatch
from src.processors.text_processor.recognizers.llm.parsers.base_parser import BaseLLMParser
from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig


def test_position_correction():
    """测试位置修正功能"""
    
    # 创建配置
    config = LLMParsersConfig()
    config.min_confidence = 0.5
    
    # 创建测试解析器
    parser = TestParser(config)
    
    # 测试文本
    original_text = "张三的手机号是13912345678,他的邮箱是zhangsan@example.com。他住在北京市海淀区中关村南大街5号，身份证号码是110101199001011234。他的信用卡号是6222020111122223333，银行账号是6225881234567890。"
    
    # 创建有位置误差的测试实体
    test_entities = [
        # 正确的实体
        EntityMatch(
            entity_type="PERSON",
            value="张三",
            start=0,
            end=2,
            confidence=1.0
        ),
        # 位置有误差的实体 - 结束位置包含了额外字符
        EntityMatch(
            entity_type="ADDRESS",
            value="海淀区中关村南大街5号",
            start=51,
            end=65,  # 实际应该是62，这里包含了"，身份"
            confidence=1.0
        ),
        # 位置有误差的实体 - 起始位置偏移
        EntityMatch(
            entity_type="ID_CARD",
            value="110101199001011234",
            start=72,  # 实际应该是69
            end=90,    # 实际应该是87
            confidence=1.0
        ),
        # 完全错误的位置
        EntityMatch(
            entity_type="PHONE_NUMBER",
            value="13912345678",
            start=100,  # 错误位置
            end=111,    # 错误位置
            confidence=1.0
        ),
    ]
    
    print("原始文本:")
    print(original_text)
    print(f"文本长度: {len(original_text)}")
    print("\n" + "="*50 + "\n")
    
    # 验证实体位置
    validated_entities = parser.validate_entity_positions(test_entities, original_text)
    
    print(f"验证结果: 原始{len(test_entities)}个实体 -> 验证通过{len(validated_entities)}个实体")
    print("\n验证通过的实体:")
    for entity in validated_entities:
        extracted_value = original_text[entity.start:entity.end]
        print(f"类型: {entity.entity_type}")
        print(f"值: '{entity.value}'")
        print(f"位置: ({entity.start}, {entity.end})")
        print(f"提取值: '{extracted_value}'")
        print(f"置信度: {entity.confidence:.2f}")
        print(f"匹配: {'✓' if extracted_value == entity.value else '✗'}")
        print("-" * 30)


if __name__ == "__main__":
    test_position_correction()
