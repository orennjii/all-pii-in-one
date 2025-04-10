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
        使用LLM生成文本响应。
        
        Args:
            prompt: 输入提示词
            temperature: 生成温度，控制随机性 (0-1)，默认为0(最确定性)
            max_tokens: 生成的最大token数
            **kwargs: 特定于实现的额外参数
            
        Returns:
            str: LLM生成的响应文本
        """
        pass
    
    @abstractmethod
    def generate_with_json_response(
        self, 
        prompt: str, 
        schema: Dict[str, Any],
        temperature: float = 0.0,
        **kwargs
    ) -> Dict[str, Any]:
        """
        使用LLM生成JSON格式的结构化响应。
        
        Args:
            prompt: 输入提示词
            schema: 预期的JSON响应结构
            temperature: 生成温度
            **kwargs: 特定于实现的额外参数
            
        Returns:
            Dict[str, Any]: 生成的结构化JSON响应
        """
        pass
    
    def batch_generate(
        self, 
        prompts: List[str],
        **kwargs
    ) -> List[str]:
        """
        批量生成LLM响应。
        
        Args:
            prompts: 提示词列表
            **kwargs: 传递给generate方法的参数
            
        Returns:
            List[str]: 生成的响应列表
        """
        responses = []
        for prompt in prompts:
            response = self.generate(prompt, **kwargs)
            responses.append(response)
        return responses
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        获取当前使用的模型信息。
        
        Returns:
            Dict[str, Any]: 包含模型信息的字典
        """
        return {
            "model_name": self.model_name,
            "provider": self.__class__.__name__.replace("Client", "")
        }