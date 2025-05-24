#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM客户端基类

定义所有LLM客户端必须实现的接口，并提供通用功能增强。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Generator, Tuple
import time
import random
import hashlib
import json
from datetime import datetime, timedelta
from collections import OrderedDict, deque

from src.configs.processors.text_processor.recognizers.llm import LLMClientConfig
from src.commons.loggers import get_module_logger

logger = get_module_logger(__name__)


class LRUCache:
    """简易LRU缓存实现"""
    
    def __init__(self, capacity: int = 1000, ttl: int = 3600):
        self.cache = OrderedDict()
        self.capacity = capacity
        self.ttl = ttl  # seconds
    
    def get(self, key: str) -> Optional[Any]:
        if key not in self.cache:
            return None
        
        value, timestamp = self.cache[key]
        if datetime.now() - timestamp > timedelta(seconds=self.ttl):
            del self.cache[key]
            return None
            
        self.cache.move_to_end(key)
        return value
    
    def put(self, key: str, value: Any) -> None:
        if key in self.cache:
            del self.cache[key]
        elif len(self.cache) >= self.capacity:
            self.cache.popitem(last=False)
            
        self.cache[key] = (value, datetime.now())


class RateLimiter:
    """令牌桶限流器"""
    
    def __init__(self, rate_limit: float, window_size: int = 60):
        """
        初始化限流器
        
        Args:
            rate_limit: 每单位时间允许的请求数
            window_size: 时间窗口大小(秒)
        """
        self.rate_limit = rate_limit
        self.window_size = window_size
        self.window = deque()
    
    def acquire(self) -> Tuple[bool, float]:
        """
        尝试获取令牌
        
        Returns:
            Tuple[bool, float]: (是否获取成功, 需要等待的时间)
        """
        now = time.time()
        
        # 清理过期的请求时间戳
        while self.window and self.window[0] <= now - self.window_size:
            self.window.popleft()
            
        # 检查是否超出限制
        if len(self.window) >= self.rate_limit:
            oldest = self.window[0]
            wait_time = oldest + self.window_size - now
            return False, wait_time
            
        self.window.append(now)
        return True, 0.0


class BaseLLMClient(ABC):
    """
    增强的LLM客户端基类
    
    所有LLM客户端实现必须继承此类并实现其抽象方法。
    提供缓存、重试、速率限制、统计信息等增强功能。
    """
    
    @abstractmethod
    def __init__(self, config: LLMClientConfig):
        """
        初始化LLM客户端
        
        Args:
            config: LLM客户端配置
        """
        self.config = config
        self.client = None
        
        # 缓存系统
        self._setup_cache()
        
        # 速率限制
        self._setup_rate_limiter()
        
        # 统计信息
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "retries": 0,
            "errors": 0,
            "total_tokens": 0,
        }
        
    def _setup_cache(self) -> None:
        """设置缓存"""
        # 默认开启缓存，容量1000，TTL 1小时
        cache_enabled = getattr(self.config, "cache_enabled", True)
        cache_ttl = getattr(self.config, "cache_ttl", 3600)
        cache_max_size = getattr(self.config, "cache_max_size", 1000)
        
        if cache_enabled:
            self.cache = LRUCache(
                capacity=cache_max_size,
                ttl=cache_ttl
            )
        else:
            self.cache = None
            
    def _setup_rate_limiter(self) -> None:
        """设置速率限制器"""
        rate_limit = getattr(self.config, "rate_limit", None)
        if rate_limit:
            self.rate_limiter = RateLimiter(rate_limit)
        else:
            self.rate_limiter = None
            
    def _get_cache_key(self, prompt: str, **kwargs) -> str:
        """
        生成缓存键
        
        Args:
            prompt: 提示词
            **kwargs: 额外参数
            
        Returns:
            str: 缓存键
        """
        # 创建包含提示词和关键参数的缓存键
        key_dict = {
            "prompt": prompt,
            "model": getattr(self, "model_name", None),
            "temperature": kwargs.get("temperature", None),
        }
        
        # 过滤掉None值
        key_dict = {k: v for k, v in key_dict.items() if v is not None}
        
        # 将字典转为字符串并哈希
        key_str = json.dumps(key_dict, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
        
    def _apply_rate_limiting(self) -> None:
        """
        应用速率限制
        
        如果达到速率限制，将等待适当的时间
        """
        if self.rate_limiter:
            allowed, wait_time = self.rate_limiter.acquire()
            if not allowed:
                logger.debug(f"速率限制: 等待 {wait_time:.2f} 秒")
                time.sleep(wait_time)
                # 重新尝试获取
                self._apply_rate_limiting()
    
    @abstractmethod
    def load(self) -> None:
        """
        加载模型或初始化API连接
        """
        pass
        
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入的提示词
            **kwargs: 额外参数
            
        Returns:
            str: 生成的文本
        """
        # 增加请求计数
        self.stats["total_requests"] += 1
        
        # 检查缓存
        if self.cache and not kwargs.get("skip_cache", False):
            cache_key = self._get_cache_key(prompt, **kwargs)
            cached_response = self.cache.get(cache_key)
            if cached_response:
                logger.debug("使用缓存的响应")
                self.stats["cache_hits"] += 1
                return cached_response
                
        # 应用速率限制
        self._apply_rate_limiting()
        
        # 具体的生成逻辑将由子类实现
        pass
        
    def generate_with_retry(self, prompt: str, **kwargs) -> str:
        """
        带重试机制的文本生成
        
        Args:
            prompt: 输入的提示词
            **kwargs: 额外参数
            
        Returns:
            str: 生成的文本
        """
        max_retries = getattr(self.config, "max_retries", 3)
        retry_delay = getattr(self.config, "retry_delay", 1.0)
        retry_backoff = getattr(self.config, "retry_backoff", 2.0)
        retry_jitter = getattr(self.config, "retry_jitter", 0.1)
        
        retries = 0
        
        while True:
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                retries += 1
                self.stats["retries"] += 1
                
                if retries >= max_retries:
                    logger.error(f"达到最大重试次数 ({max_retries})，放弃重试")
                    self.stats["errors"] += 1
                    raise
                
                # 计算下次重试的延迟
                jitter = random.uniform(-retry_jitter, retry_jitter) * retry_delay
                actual_delay = retry_delay + jitter
                
                logger.warning(f"生成失败: {str(e)}. 将在 {actual_delay:.2f} 秒后重试 (尝试 {retries}/{max_retries})")
                time.sleep(actual_delay)
                
                # 增加重试延迟
                retry_delay *= retry_backoff
    
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """
        批量生成文本
        
        Args:
            prompts: 输入的提示词列表
            **kwargs: 额外参数
            
        Returns:
            List[str]: 生成的文本列表
        """
        return [self.generate_with_retry(prompt, **kwargs) for prompt in prompts]
        
    def generate_stream(self, prompt: str, **kwargs) -> Generator[str, None, None]:
        """
        流式生成文本
        
        Args:
            prompt: 输入的提示词
            **kwargs: 额外参数
            
        Returns:
            Generator[str, None, None]: 生成的文本流
        """
        # 默认实现，将由支持流式生成的子类覆盖
        logger.warning("当前客户端不支持流式生成，使用普通生成代替")
        yield self.generate_with_retry(prompt, **kwargs)
        
    def get_stats(self) -> Dict[str, Any]:
        """
        获取使用统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        return self.stats
        
    def reset_stats(self) -> None:
        """重置统计信息"""
        self.stats = {k: 0 for k in self.stats}
        
    @abstractmethod
    def is_available(self) -> bool:
        """
        检查客户端是否可用
        
        Returns:
            bool: 如果客户端可用，返回True，否则返回False
        """
        pass
