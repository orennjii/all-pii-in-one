"""
LLM 响应解析器基类模块。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from presidio_analyzer import RecognizerResult


class BaseParser(ABC):
    """
    LLM输出解析器的抽象基类。
    
    定义了将LLM输出解析为标准格式的接口, 用于提取识别的实体及其在原始文本中的位置。
    所有具体的解析器实现都应该继承这个类并实现指定的方法。
    """

    def __init__(self, **kwargs):
        """
        初始化解析器。
        
        Args:
            **kwargs: 解析器特定的参数
        """
        self.kwargs = kwargs
    
    @abstractmethod
    def parse(
        self, 
        text: str,
        llm_response: str, 
        entities: Optional[List[str]] = None
    ) -> List[RecognizerResult]:
        """
        解析LLM响应并返回识别器结果列表。
        
        Args:
            text: 原始输入文本
            llm_response: LLM返回的原始文本响应
            entities: 要查找的实体类型列表（可选）
            
        Returns:
            List[RecognizerResult]: 包含实体识别结果的列表
                
        Raises:
            ValueError: 如果无法解析LLM响应
        """
        pass
    
    def _find_text_position(self, original_text: str, entity_text: str) -> tuple:
        """
        在原始文本中查找实体文本的位置。
        
        Args:
            original_text: 原始文本
            entity_text: 要查找的实体文本
            
        Returns:
            tuple: (start, end) 位置元组
            
        Raises:
            ValueError: 如果在原始文本中找不到实体文本
        """
        start = original_text.find(entity_text)
        if start == -1:
            raise ValueError(f"在原文中找不到实体文本: {entity_text}")
            
        end = start + len(entity_text)
        return (start, end)
