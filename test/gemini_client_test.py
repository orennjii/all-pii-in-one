#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini客户端测试脚本

该脚本演示如何使用AppConfig从配置文件加载配置，并初始化GeminiClient
"""

import os
import sys
import logging
import time
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.commons import get_module_logger, find_project_root
from src.configs import AppConfig
from src.processors.text_processor.recognizers.llm.clients.gemini_client import GeminiClient
from src.processors.text_processor.recognizers.llm.clients.client_factory import create_llm_client

# 获取模块日志记录器
logger = get_module_logger(__name__)

def main():
    """
    主函数 - 演示如何使用AppConfig加载配置并初始化GeminiClient
    """
    # 设置环境变量（如果需要通过代理访问API）
    try:
        # 如果需要代理，请取消下面两行的注释并设置正确的代理地址
        # os.environ["HTTP_PROXY"] = "http://127.0.0.1:7890"
        # os.environ["HTTPS_PROXY"] = "http://127.0.0.1:7890"
        logger.info("环境变量设置完成")
    except Exception as e:
        logger.warning(f"设置环境变量出错: {e}")

    # 从配置文件加载应用配置
    config_path = find_project_root(Path(__file__)) / "config" / "app_config.yaml"
    logger.info(f"正在从 {config_path} 加载配置")
    
    app_config = AppConfig.load_from_yaml(str(config_path))
    logger.info("配置加载成功")
    
    # 获取文本处理器中的LLM客户端配置
    text_processor_config = app_config.text_processor
    recognizers_config = text_processor_config.recognizers
    llm_config = recognizers_config.llm
    
    # 打印当前配置信息
    logger.info(f"LLM识别器启用状态: {llm_config.enabled}")
    logger.info(f"LLM客户端类型: {llm_config.client.type}")
    logger.info(f"LLM模型: {llm_config.client.model_name_or_path}")
    
    # 修改配置以使用Gemini
    llm_client_config = llm_config.client
    
    # 更新配置以使用Gemini (如果原配置不是Gemini)
    if llm_client_config.type != "Gemini":
        logger.info("更新配置以使用Gemini客户端")
        # 创建一个新的配置副本并更新
        updated_config = llm_client_config.update({
            "type": "Gemini",
            "model_name_or_path": "gemini-2.5-flash-preview-05-20",
            "temperatures": 0.7,
            "max_tokens": 2000,
            "api_key": os.environ.get("GEMINI_API_KEY", "")  # 从环境变量获取API密钥
        })
    else:
        updated_config = llm_client_config
    
    # 方法1：通过GeminiClient类直接初始化
    try:
        logger.info("方法1: 直接使用GeminiClient类初始化客户端")
        client = GeminiClient(updated_config)
        client.load()
        logger.info("GeminiClient初始化成功")
        
        # 测试生成文本
        prompt = "请简要介绍一下什么是个人隐私信息(PII)以及为什么保护它很重要？"
        logger.info(f"发送提示词: {prompt}")
        
        response = client.generate(prompt)
        logger.info(f"Gemini响应:\n{response}\n")

        # 测试流式生成
        logger.info("开始流式生成示例...")
        response_stream = client.generate_stream(prompt)
        print("", end="", flush=True)  # 初始化打印
        for text_chunk in response_stream:
            print(text_chunk, end="", flush=True)
        
        # 完成后打印换行
        print("\n")
        print("-" * 50)
        print("\n流式生成完成！")
        
    except Exception as e:
        logger.error(f"使用GeminiClient初始化失败: {str(e)}")
    
    # # 方法2：通过工厂函数创建客户端
    # try:
    #     logger.info("\n方法2: 使用工厂函数创建LLM客户端")
    #     factory_client = create_llm_client(updated_config)
    #     logger.info(f"创建的客户端类型: {factory_client.__class__.__name__}")
        
    #     # 测试生成文本
    #     prompt = "请列举5种常见的个人隐私信息类型及其保护方法。"
    #     logger.info(f"发送提示词: {prompt}")
        
    #     response = factory_client.generate(prompt)
    #     logger.info(f"客户端响应:\n{response}")
        
    # except Exception as e:
    #     logger.error(f"使用工厂函数创建客户端失败: {str(e)}")

if __name__ == "__main__":
    main()
