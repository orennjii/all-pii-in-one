#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Google Gemini LLM客户端实现

使用最新的google-genai SDK库实现与Gemini API的交互。
"""

import os
import time
from typing import Dict, Any, Optional, Generator, List

# 导入所需的包
from google import genai
from google.genai import types

from src.configs.processors.text_processor.recognizers.llm import LLMClientConfig
from src.commons.loggers import get_module_logger
from src.processors.text_processor.recognizers.llm.clients.base_client import BaseLLMClient

logger = get_module_logger(__name__)


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API 客户端实现
    
    通过 Google Gemini API 接口调用 Gemini 大语言模型。
    使用最新的 google-genai SDK。
    """
    
    def __init__(
        self,
        config: LLMClientConfig,
        system_prompt: Optional[str] = None,
        **kwargs: Any
    ):
        """
        初始化Gemini客户端
        
        Args:
            config: LLM客户端配置
        """
        super().__init__(config)
        self.use_vertex = config.use_vertex or False
        self.api_key = os.environ.get("GEMINI_API_KEY") or config.api_key
        self.endpoint = config.endpoint
        self.model_name = config.model_name_or_path

        self.temperature = config.temperatures
        self.max_tokens = config.max_tokens
        self.top_k = config.top_k
        self.top_p = config.top_p

        self.system_prompt = system_prompt or config.system_prompt

        self.client: Optional[genai.Client] = None

    def load(self) -> None:
        """
        初始化Gemini客户端
        
        Raises:
            ImportError: 如果缺少google-genai包
            ValueError: 如果缺少API密钥
        """
        if not self.api_key:
            logger.error("未指定Google API密钥")
            raise ValueError("Google API密钥未指定，请在配置中设置或通过环境变量GEMINI_API_KEY提供")
        
        # 配置Gemini客户端
        try:
            # 配置API密钥
            self.client = genai.Client(
                api_key=self.api_key
            )
                
            logger.info(f"Gemini客户端初始化成功")
                
        except Exception as e:
            logger.error(f"Gemini客户端初始化失败: {str(e)}")
            raise
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入的提示词
            **kwargs: 额外参数，可能包括max_tokens, temperature等
            
        Returns:
            str: 生成的文本
            
        Raises:
            RuntimeError: 如果客户端未初始化或生成失败
        """
        # 调用父类方法处理缓存和统计
        super().generate(prompt, **kwargs)
        
        if not self.client:
            self.load()
        
        try:
            # 设置参数
            temperature = kwargs.get("temperature", self.temperature)
            top_p = kwargs.get("top_p", self.top_p)
            max_tokens = kwargs.get("max_tokens", self.max_tokens)
            system_prompt = kwargs.get("system_prompt", self.system_prompt)
            
            # 创建生成配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=top_p,
            )
            
            # 处理系统提示词
            if system_prompt:
                config.system_instruction = system_prompt
            
            # 生成响应
            assert self.client is not None
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # 获取响应文本
            response_text = response.text
            if response_text is None:
                logger.error("Gemini生成的文本为空")
                raise RuntimeError("LLM生成的文本为空")
            logger.debug(f"Gemini生成的文本: {response_text}")

            # 将结果存入缓存
            if self.cache and not kwargs.get("skip_cache", False):
                cache_key = self._get_cache_key(prompt, **kwargs)
                self.cache.put(cache_key, response_text)
                
            return response_text
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Gemini生成文本失败: {str(e)}")
            raise RuntimeError(f"LLM生成失败: {str(e)}")

    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入的提示词
            **kwargs: 额外参数
            
        Returns:
            Generator[str, None, None]: 生成的文本流
        """
        if not self.client:
            self.load()
            
        # 应用速率限制
        self._apply_rate_limiting()
        
        self.stats["total_requests"] += 1
        
        try:
            # 设置参数
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", 1000)
            system_prompt = kwargs.get("system_prompt", self.system_prompt)
            
            # 创建生成配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 0)
            )
            
            # 处理系统提示词
            if system_prompt:
                config.system_instruction = system_prompt
            
            assert self.client is not None
            stream_response = self.client.models.generate_content_stream(
                model=self.model_name,
                contents=prompt,
                config=config
            )
            
            # 流式返回结果
            full_response = ""
            for chunk in stream_response:
                if hasattr(chunk, 'text') and chunk.text:
                    full_response += chunk.text
                    yield chunk.text
            
            # 结束后，将完整响应存入缓存
            if self.cache and not kwargs.get("skip_cache", False):
                cache_key = self._get_cache_key(prompt, **kwargs)
                self.cache.put(cache_key, full_response)
                
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Gemini流式生成失败: {str(e)}")
            raise RuntimeError(f"LLM流式生成失败: {str(e)}")
            
    def generate_multimodal(self, text_prompt: str, image_data: List[bytes], **kwargs) -> str:
        """
        多模态生成（文本+图像）
        
        Args:
            text_prompt: 文本提示词
            image_data: 图像数据列表（bytes格式）
            **kwargs: 额外参数
            
        Returns:
            str: 生成的文本
        """
        if not self.client:
            self.load()
        
        # 检查模型是否支持多模态
        use_vision_model = True
        original_model_name = self.model_name
        vision_model_name = "gemini-pro-vision"
        
        if not self.model_name.endswith("-vision") and self.model_name != "gemini-pro-vision":
            logger.warning(f"当前模型 {self.model_name} 可能不支持多模态输入，尝试使用 {vision_model_name}")
            use_vision_model = True
            
        try:
            # 应用速率限制
            self._apply_rate_limiting()
            
            self.stats["total_requests"] += 1
            
            # 设置参数
            temperature = kwargs.get("temperature", self.temperature)
            max_tokens = kwargs.get("max_tokens", 1000)
            
            # 创建多模态内容 - 使用正确的API格式
            parts = [types.Part.from_text(text=text_prompt)]
                
            # 添加图像
            for img_data in image_data:
                # 使用 Part.from_bytes 而不是 from_image
                image_part = types.Part.from_bytes(data=img_data, mime_type="image/jpeg")
                parts.append(image_part)
            
            # 将parts组合为内容
            contents = parts  # type: ignore # API支持Part列表，但类型检查器不识别
            
            # 创建生成配置
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                top_p=kwargs.get("top_p", 0.95),
                top_k=kwargs.get("top_k", 0)
            )
            
            # 使用适合的模型生成响应
            model_name = vision_model_name if use_vision_model else self.model_name
            
            # 生成响应
            assert self.client is not None
            response = self.client.models.generate_content(
                model=model_name,
                contents=contents,  # type: ignore # API支持Part列表，但类型检查器不识别
                config=config
            )
            
            # 获取响应文本
            response_text = response.text
            if response_text is None:
                logger.error("Gemini多模态生成的文本为空")
                raise RuntimeError("LLM多模态生成的文本为空")
            
            return response_text
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Gemini多模态生成失败: {str(e)}")
            raise RuntimeError(f"LLM多模态生成失败: {str(e)}")
                
    def generate_text_embeddings(self, texts: List[str], **kwargs) -> List[List[float]]:
        """
        生成文本嵌入向量
        
        Args:
            texts: 要生成嵌入的文本列表
            **kwargs: 额外参数
            
        Returns:
            List[List[float]]: 嵌入向量列表
        """
        if not self.client:
            self.load()
            
        # 应用速率限制
        self._apply_rate_limiting()
        
        self.stats["total_requests"] += 1
        
        try:
            # 创建嵌入配置
            config = types.EmbedContentConfig()
            if kwargs.get("dimensions"):
                config.output_dimensionality = kwargs.get("dimensions")
                
            # 生成嵌入
            embedding_model = "text-embedding-004"  # 使用新的embedding模型
                
            assert self.client is not None
            response = self.client.models.embed_content(
                model=embedding_model, 
                contents=texts,  # type: ignore # 直接传递文本列表，API支持但类型检查器不识别
                config=config
            )
            
            # 提取嵌入向量 - 使用正确的字段名
            embeddings = []
            if response.embeddings:
                for embedding in response.embeddings:
                    if hasattr(embedding, 'values') and embedding.values:
                        embeddings.append(embedding.values)
                    else:
                        raise RuntimeError("嵌入向量格式错误")
            else:
                raise RuntimeError("无法获取嵌入向量")
            
            return embeddings
            
        except Exception as e:
            self.stats["errors"] += 1
            logger.error(f"Gemini嵌入生成失败: {str(e)}")
            raise RuntimeError(f"嵌入生成失败: {str(e)}")
            
    def is_available(self) -> bool:
        """
        检查客户端是否可用
        
        Returns:
            bool: 如果客户端可用，返回True，否则返回False
        """
        if not self.api_key:
            return False
            
        try:
            if not self.client:
                self.load()
            
            # 尝试列出可用模型
            assert self.client is not None
            list(self.client.models.list())
            return True
        except Exception as e:
            logger.error(f"Gemini客户端不可用: {str(e)}")
            return False
