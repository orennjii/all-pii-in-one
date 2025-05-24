#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM配置和客户端使用示例
"""

from src.configs import LLM_CONFIG
from src.processors.text_processor.recognizers.llm.clients import OpenAIClient, GeminiClient, AnthropicClient


def show_config_usage():
    """展示配置系统的使用"""
    
    print("=== LLM配置使用示例 ===")
    
    # 1. 获取全局配置单例
    print("\n1. 访问全局配置")
    print(f"当前默认LLM提供商: {LLM_CONFIG.default_provider}")
    
    # 2. 获取特定提供商的配置
    print("\n2. 获取特定提供商配置")
    openai_config = LLM_CONFIG.get_provider_config("openai")
    print(f"OpenAI配置 - 模型: {openai_config.model}, 温度: {openai_config.temperature}")
    
    gemini_config = LLM_CONFIG.get_provider_config("gemini")
    print(f"Gemini配置 - 模型: {gemini_config.model}, 温度: {gemini_config.temperature}")
    
    # 3. 获取默认提供商的配置
    print("\n3. 获取默认提供商配置")
    default_config = LLM_CONFIG.get_provider_config()
    print(f"默认配置 - 提供商: {LLM_CONFIG.default_provider}, 模型: {default_config.model}")
    
    # 4. 演示配置继承
    print("\n4. 配置继承机制")
    print(f"基础超时设置: {LLM_CONFIG.base.timeout}秒")
    print(f"OpenAI继承的超时设置: {openai_config.timeout}秒")
    
    # 5. 配置结构
    print("\n5. 配置结构")
    print("LLM_CONFIG")
    print("|-- base (基础配置)")
    print("|   |-- api_key")
    print("|   |-- api_base")
    print("|   |-- timeout")
    print("|   |-- retry_count")
    print("|   `-- retry_delay")
    print("|")
    print("|-- openai (OpenAI配置)")
    print("|   |-- [继承的基础配置]")
    print("|   |-- model")
    print("|   |-- max_tokens")
    print("|   |-- temperature")
    print("|   `-- [其他特定参数]")
    print("|")
    print("|-- gemini (Google Gemini配置)")
    print("|   |-- [继承的基础配置]")
    print("|   |-- model")
    print("|   |-- max_output_tokens")
    print("|   `-- [其他特定参数]")
    print("|")
    print("`-- anthropic (Anthropic配置)")
    print("    |-- [继承的基础配置]")
    print("    |-- model")
    print("    |-- max_tokens")
    print("    `-- [其他特定参数]")


def show_client_usage():
    """展示客户端的使用"""
    
    print("\n=== LLM客户端使用示例 ===")
    
    print("\n1. 使用默认配置创建客户端")
    print("client = OpenAIClient()")
    print("# 客户端会自动从LLM_CONFIG加载openai配置")
    
    print("\n2. 覆盖部分配置")
    print("client = OpenAIClient(model='gpt-3.5-turbo', temperature=0.5)")
    print("# 使用指定的模型和温度，其他参数从配置加载")
    
    print("\n3. 使用全局配置中的不同提供商")
    print("# 获取当前默认提供商")
    print(f"默认提供商: {LLM_CONFIG.default_provider}")
    print("\n# 临时修改默认提供商（不推荐）")
    print("LLM_CONFIG.default_provider = 'gemini'")
    print("client = GeminiClient()")
    
    print("\n4. 发送提示词")
    print("response = client.send_prompt('分析以下文本中的个人敏感信息: \"张三的身份证号是330102198801010101\"')")
    print("result = response['text']")
    
    print("\n5. 错误处理")
    print("try:")
    print("    response = client.send_prompt(prompt)")
    print("except Exception as e:")
    print("    print(f'LLM请求失败: {str(e)}')")


if __name__ == "__main__":
    show_config_usage()
    show_client_usage()
