#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini客户端PII检测测试脚本

该脚本演示如何使用AppConfig从配置文件加载配置，并使用GeminiClient进行PII检测
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.append(str(Path(__file__).parent.parent))

from src.commons.loggers import get_module_logger
from src.configs import AppConfig
from src.processors.text_processor.recognizers.llm.clients.gemini_client import GeminiClient
from src.processors.text_processor.recognizers.llm.parsers.gemini_parser import GeminiResponseParser

# 获取模块日志记录器
logger = get_module_logger(__name__)

def main():
    """
    主函数 - 演示如何使用GeminiClient进行PII检测
    """
    # 从配置文件加载应用配置
    config_path = Path(__file__).parent.parent / "config" / "app_config.yaml"
    logger.info(f"正在从 {config_path} 加载配置")
    
    app_config = AppConfig.load_from_yaml(str(config_path))
    logger.info("配置加载成功")
    
    # 获取文本处理器中的LLM客户端配置
    llm_config = app_config.text_processor.recognizers.llm.client
    
    # 设置API密钥
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = input("请输入Gemini API密钥: ")
        if not api_key:
            logger.error("未提供API密钥，无法继续")
            return
    
    # 更新配置以使用Gemini
    client_config = llm_config.update({
        "type": "Gemini",
        "model_name_or_path": "gemini-2.5-flash-preview-05-20",
        "temperatures": 0.2,  # 降低温度以获得更确定性的输出
        "api_key": api_key
    })
    
    # 初始化GeminiClient
    try:
        client = GeminiClient(client_config)
        client.load()
        logger.info("GeminiClient初始化成功")
        
        # 初始化解析器
        parser = GeminiResponseParser()
        
        # 测试文本
        test_text = """
        这是一份包含敏感信息的示例文本：

        我的姓名是张三，来自北京市海淀区。
        我的车牌号是京A12345，手机号码是13812345678。
        我的身份证号是110102199003074258，银行卡号是6225880137998765。

        2023年10月15日，我访问了www.example.com并发送邮件到test@example.com，IP地址是192.168.1.1。
        """
        
        # 构建PII检测提示词
        prompt = f"""
        请帮我找出以下文本中的所有个人隐私信息(PII)，以JSON格式返回结果。
        每个找到的实体应包含实体类型、文本内容和在原文中的位置信息。
        
        请严格按照以下格式返回JSON:
        [
          {{"entity_type": "PERSON", "text": "实体文本", "start": 开始位置, "end": 结束位置, "score": 0.95}},
          ...
        ]
        
        需要检测的实体类型包括:
        PERSON(人名), ID_CARD(身份证), PHONE_NUMBER(电话号码), EMAIL_ADDRESS(电子邮件), 
        CREDIT_CARD(信用卡), BANK_ACCOUNT(银行账号), ADDRESS(地址), LOCATION(地点), 
        URL(网址), IP_ADDRESS(IP地址), DATE_TIME(日期时间), CAR_PLATE(车牌)
        
        原始文本:
        {test_text}
        """
        
        logger.info("发送PII检测请求...")
        response = client.generate(prompt)
        logger.info(f"原始响应:\n{response}")
        
        # 尝试解析响应
        try:
            results = parser.parse(test_text, response)
            logger.info(f"检测到 {len(results)} 个PII实体:")
            
            for result in results:
                logger.info(f"类型: {result.entity_type}, 文本: {result.text}, " 
                          f"位置: {result.start}-{result.end}, 得分: {result.score}")
        except Exception as e:
            logger.error(f"解析响应失败: {str(e)}")
    
    except Exception as e:
        logger.error(f"GeminiClient使用失败: {str(e)}")

if __name__ == "__main__":
    main()
