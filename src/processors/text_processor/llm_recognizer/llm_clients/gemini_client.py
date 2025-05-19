import logging
import time
import json
from typing import Dict, Any, Optional, List, Union
import pprint
import os

import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from tenacity import retry, stop_after_attempt, wait_exponential

from .base_client import BaseLLMClient

logger = logging.getLogger("llm_recognizer.gemini_client")

try:
    os.environ["HTTP_PROXY"] = "http://127.0.0.1:17072"
    os.environ["HTTPS_PROXY"] = "http://127.0.0.1:17072"
    logger.info("已成功设置代理")
except Exception as e:
    logger.warning(f"Error setting environment variables: {e}")


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API的客户端实现。
    
    使用Google的Generative AI Python库与Gemini模型交互。
    """
    
    def __init__(
        self,
        model_name: str = 'models/gemini-2.0-flash-thinking-exp-01-21',
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        safety_settings: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """
        初始化Gemini客户端。
        
        Args:
            model_name: Gemini模型名称, 默认为'models/gemini-2.0-flash-thinking-exp-01-21'
            api_key: Google API密钥
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            safety_settings: Gemini安全设置配置
            **kwargs: 其他Gemini特定参数
        """
        self.safety_settings = safety_settings
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
    
    def _setup_client(self) -> None:
        """设置Gemini API客户端"""
        if not self.api_key:
            raise ValueError("使用Gemini API需要提供API密钥")
        
        # 配置Gemini API
        print("正在配置Gemini API客户端...")
        genai.configure(api_key=self.api_key)
        
        # 验证模型是否可用
        try:
            print("正在验证模型可用性...")
            models = genai.list_models()
            
            available_models = [model.name for model in models]
            
            if self.model_name not in available_models:
                logger.warning(f"模型 {self.model_name} 可能不可用。可用模型: {available_models}")
        except Exception as e:
            logger.warning(f"无法验证模型可用性: {e}")
        
        logger.info(f"Gemini客户端设置完成，使用模型: {self.model_name}")
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        使用Gemini生成文本响应。
        
        Args:
            prompt: 输入提示词
            temperature: 生成温度 (0-1)
            max_tokens: 生成的最大token数
            top_p: nucleus采样参数
            top_k: top-k采样参数
            **kwargs: 额外的生成参数
            
        Returns:
            str: 生成的文本响应
        """
        start_time = time.time()
        
        try:
            # 创建生成配置
            generation_config = GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
                top_k=top_k
            )
            
            # 获取模型实例
            model = genai.GenerativeModel(
                model_name=self.model_name,
                generation_config=generation_config,
                safety_settings=self.safety_settings
            )
            
            # 生成响应
            response = model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                result = response.text
            else:
                # 处理可能的其他响应格式
                result = str(response)
            
            elapsed_time = time.time() - start_time
            logger.debug(f"Gemini生成完成，耗时: {elapsed_time:.2f}秒")
            
            return result
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"Gemini生成失败，耗时: {elapsed_time:.2f}秒，错误: {str(e)}")
            raise
    
    def generate_with_json_response(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用Gemini生成符合指定schema的JSON响应。
        
        Args:
            prompt: 输入提示词
            schema: 预期的JSON响应结构
            temperature: 生成温度
            **kwargs: 额外的生成参数
            
        Returns:
            Dict[str, Any]: 生成的JSON响应
        """
        # 构建带有明确JSON指令的提示词
        json_prompt = (
            f"{prompt}\n\n"
            f"请以JSON格式返回响应，遵循以下模式:\n"
            f"{json.dumps(schema, indent=2, ensure_ascii=False)}\n\n"
            f"仅返回有效的JSON，不要包含额外解释。"
        )
        
        # 生成JSON响应
        response_text = self.generate(
            prompt=json_prompt, 
            temperature=temperature,
            **kwargs
        )
        
        # 解析JSON响应
        try:
            # 尝试从响应文本中提取JSON部分
            json_str = self._extract_json(response_text)
            result = json.loads(json_str)
            return result
        except json.JSONDecodeError as e:
            logger.error(f"无法解析Gemini返回的JSON: {e}")
            logger.debug(f"原始响应: {response_text}")
            raise ValueError(f"无法从Gemini响应中解析JSON: {str(e)}")
    
    def _extract_json(self, text: str) -> str:
        """
        从文本中提取JSON部分。
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            str: 提取的JSON字符串
        """
        # 尝试查找JSON代码块 (```json ... ```)
        import re
        json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_block_match:
            return json_block_match.group(1)
        
        # 尝试查找开始和结束大括号之间的内容
        json_match = re.search(r'(\{[\s\S]*\})', text)
        if json_match:
            return json_match.group(1)
        
        # 如果找不到明确的JSON结构，返回原始文本
        return text