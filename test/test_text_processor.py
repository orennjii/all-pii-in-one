#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TextProcessor 测试模块

该模块演示如何从app_config.yaml加载配置并使用TextProcessor类处理文本中的个人隐私信息
可以通过以下方式运行:
python test/test_text_processor.py
或
PYTHONPATH=. python -m test.test_text_processor
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from src.commons.utils import find_project_root
from src.configs import AppConfig
from src.processors.text_processor import TextProcessor
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)

current_path = Path(__file__).resolve()
project_root = find_project_root(current_path)

def test_text_processor():
    """
    测试TextProcessor的主函数
    
    从app_config.yaml加载配置，并使用TextProcessor类处理文本中的个人隐私信息
    """
    
    # 从app_config.yaml加载配置
    config_path = project_root / "config" / "app_config.yaml"
    app_config = AppConfig.load_from_yaml(str(config_path))
    
    # 获取文本处理器配置
    text_processor_config = app_config.processor.text_processor
    
    # 创建TextProcessor实例
    text_processor = TextProcessor(config=text_processor_config)
    
    # 测试文本
    test_text = "张三的手机号是13912345678,他的邮箱是zhangsan@example.com。他住在北京市海淀区中关村南大街5号，身份证号码是110101199001011234。他的信用卡号是6222020111122223333，银行账号是6225881234567890。"
    
    logger.info("开始使用TextProcessor处理文本...")
    
    # 测试完整的处理流程
    logger.info("=" * 50)
    logger.info("测试1: 完整处理流程(分析+匿名化)")
    logger.info("=" * 50)
    
    processing_result = text_processor.process(
        text=test_text,
        enable_segmentation=False,  # 不启用分割
        enable_analysis=True,       # 启用分析
        enable_anonymization=True,  # 启用匿名化
        language='zh'
    )
    
    logger.info(f"原始文本: {processing_result.original_text}")
    logger.info(f"匿名化后文本: {processing_result.anonymized_text}")
    logger.info(f"发现的段落数: {len(processing_result.segments)}")
    
    # 统计所有检测到的实体
    total_entities = sum(len(results) for results in processing_result.analysis_results)
    logger.info(f"检测到的PII实体总数: {total_entities}")
    
    # 详细显示检测结果
    for i, segment_results in enumerate(processing_result.analysis_results):
        if segment_results:
            logger.info(f"段落 {i+1} 中检测到 {len(segment_results)} 个PII实体:")
            for result in segment_results:
                segment_text = processing_result.segments[i].text if processing_result.segments else test_text
                entity_value = segment_text[result.start:result.end]
                logger.info(f"  类型: {result.entity_type}, 值: {entity_value}, "
                          f"位置: {result.start}-{result.end}, 置信度: {result.score:.2f}")
    
    # 测试仅分析功能
    logger.info("\n" + "=" * 50)
    logger.info("测试2: 仅分析功能")
    logger.info("=" * 50)
    
    analysis_results = text_processor.analyze_only(
        text=test_text,
        language='zh'
    )
    
    logger.info(f"仅分析模式检测到 {len(analysis_results)} 个PII实体:")
    for result in analysis_results:
        entity_value = test_text[result.start:result.end]
        logger.info(f"  类型: {result.entity_type}, 值: {entity_value}, "
                  f"位置: {result.start}-{result.end}, 置信度: {result.score:.2f}")
    
    # 测试仅匿名化功能（使用前面的分析结果）
    logger.info("\n" + "=" * 50)
    logger.info("测试3: 仅匿名化功能")
    logger.info("=" * 50)
    
    anonymize_result = text_processor.anonymize_only(
        text=test_text,
        analyzer_results=analysis_results
    )
    
    logger.info(f"仅匿名化模式结果: {anonymize_result.text}")
    logger.info(f"匿名化操作数: {len(anonymize_result.items)}")
    
    # 测试文本分割功能
    logger.info("\n" + "=" * 50)
    logger.info("测试4: 文本分割功能")
    logger.info("=" * 50)
    
    segments = text_processor.segment_only(test_text)
    logger.info(f"文本分割成 {len(segments)} 个段落:")
    for i, segment in enumerate(segments):
        logger.info(f"  段落 {i+1}: {segment.text[:50]}{'...' if len(segment.text) > 50 else ''}")
    
    # 测试获取支持的实体类型
    logger.info("\n" + "=" * 50)
    logger.info("测试5: 获取支持的功能")
    logger.info("=" * 50)
    
    supported_entities = text_processor.get_supported_entities()
    logger.info(f"支持的实体类型: {supported_entities}")
    
    supported_operators = text_processor.get_supported_operators()
    logger.info(f"支持的匿名化操作符: {supported_operators}")
    
    # 测试带分割的完整处理
    logger.info("\n" + "=" * 50)
    logger.info("测试6: 带分割的完整处理")
    logger.info("=" * 50)
    
    # 使用较长的测试文本，适合分割
    long_test_text = """
    张三是一名软件工程师，他的个人信息如下。
    
    联系方式：手机号13912345678，邮箱zhangsan@example.com。
    
    住址：北京市海淀区中关村南大街5号。
    
    证件信息：身份证号码110101199001011234。
    
    金融信息：信用卡号6222020111122223333，银行账号6225881234567890。
    """
    
    segmented_result = text_processor.process(
        text=long_test_text,
        enable_segmentation=True,   # 启用分割
        enable_analysis=True,       # 启用分析
        enable_anonymization=True,  # 启用匿名化
        language='zh'
    )
    
    logger.info(f"分割后的段落数: {len(segmented_result.segments)}")
    logger.info(f"匿名化后的完整文本: {segmented_result.anonymized_text}")
    
    # 显示每个段落的处理结果
    for i, (segment, segment_analysis, segment_anonymized) in enumerate(
        zip(segmented_result.segments, segmented_result.analysis_results, segmented_result.anonymized_segments)
    ):
        logger.info(f"\n段落 {i+1}:")
        logger.info(f"  原始: {segment.text.strip()}")
        logger.info(f"  匿名化: {segment_anonymized.text.strip()}")
        logger.info(f"  检测到实体数: {len(segment_analysis)}")
    
    logger.info("\n" + "=" * 50)
    logger.info("TextProcessor 测试完成!")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_text_processor()