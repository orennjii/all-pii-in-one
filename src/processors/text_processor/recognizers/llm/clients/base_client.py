"""
LLM 客户端基类模块。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union


class BaseLLMClient(ABC):
    """
    大语言模型客户端的抽象基类。
    
    定义了所有LLM客户端实现必须提供的接口，用于与不同的大语言模型服务进行交互。
    """
    
    def __init__(
        self, 
        model_name: str,
        api_key: Optional[str] = None,
        timeout: int = 60,
        max_retries: int = 3,
        **kwargs
    ):
        """
        初始化LLM客户端。
        
        Args:
            model_name: 要使用的模型名称
            api_key: API密钥 (如果需要)
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            **kwargs: 额外的特定于实现的参数
        """
        self.model_name = model_name
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.kwargs = kwargs
        self._setup_client()
    
    @abstractmethod
    def _setup_client(self) -> None:
        """
        设置与特定LLM提供商API交互的客户端。
        每个具体实现都必须提供此方法的实现。
        """
        pass
    
    @abstractmethod
    def generate(
        self, 
        prompt: str, 
        temperature: float = 0.0,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> str:
        """
        生成LLM响应。
        
        Args:
            prompt: 发送到LLM的提示文本
            temperature: 控制随机性的温度参数 (0.0表示确定性输出)
            max_tokens: 生成的最大标记数
            **kwargs: 其他特定于实现的参数
            
        Returns:
            str: LLM生成的响应文本
            
        Raises:
            Exception: 如果生成过程中出现任何错误
        """
        pass
    
    @abstractmethod
    def generate_with_json_output(
        self,
        prompt: str,
        json_schema: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        生成JSON格式的LLM响应。
        
        Args:
            prompt: 发送到LLM的提示文本
            json_schema: 期望输出的JSON模式
            temperature: 控制随机性的温度参数
            **kwargs: 其他特定于实现的参数
            
        Returns:
            Dict[str, Any]: 解析为字典的JSON响应
            
        Raises:
            Exception: 如果生成或解析过程中出现任何错误
        """
        pass
