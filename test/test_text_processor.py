#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TextProcessor 测试模块

该模块演示如何从app_config.yaml加载配置并使用TextProcessor类处理文本中的个人隐私信息
可以通过以下方式运行:
PYTHONPATH=. python test/test_text_processor.py
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

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
    test_text = """
尊敬的客服团队：

你们好，我叫李明轩，想咨询一下我的订单 #TS20250527A88G 的最新发货状态。这个订单是我上周二下午通过你们的手机应用下的，当时预估3天内发货，但现在还没有收到任何更新。

我的注册手机号是 138-1088-6688，注册邮箱是 mingxuan.li.test@emailservice.cn。如果需要核实身份，我的会员卡号是 VIP9876543210。

麻烦你们帮忙查一下，看看包裹具体到哪里了。如果已经发出了，希望能提供一下快递单号。我的收货地址是：北京市海淀区中关村南大街28号院3号楼B座1101室，邮编100081。

另外，我记得当时购买这款“智能空气净化器Pro”时，客服代表王小姐（工号大概是A073）提到，如果我是1990年6月15日之前出生的，可以享受一个额外的老客户折扣。我的出生日期是1988年10月26日，不知道这个折扣是否已经应用到订单里了？如果方便的话，也请一并核实。

非常感谢！期待你们的回复。

祝好
"""
    
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
    
    logger.info("\n" + "=" * 50)
    logger.info("TextProcessor 测试完成!")
    logger.info("=" * 50)

if __name__ == "__main__":
    test_text_processor()