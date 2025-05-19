from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union

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
        llm_response: str, 
        original_text: str, 
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析LLM响应并返回标准化的实体信息列表。
        
        Args:
            llm_response: LLM返回的原始文本响应
            original_text: 发送给LLM的原始输入文本
            entities: 要查找的实体类型列表（可选）
            
        Returns:
            List[Dict[str, Any]]: 包含实体信息的字典列表，每个字典至少应包含:
                - entity_type: 实体类型 (如 "PERSON", "EMAIL")
                - start: 实体在原始文本中的起始位置
                - end: 实体在原始文本中的结束位置
                - score: 置信度分数 (可选)
                - value: 提取的实体值 (可选)
        """
        pass
    
    def _find_text_position(self, original_text: str, entity_text: str) -> tuple:
        """
        在原始文本中查找实体文本的位置。
        
        Args:
            original_text: 原始文本
            entity_text: 要查找的实体文本
            
        Returns:
            tuple: (start, end) 在原始文本中的位置，如果未找到则返回 (-1, -1)
        """
        start = original_text.find(entity_text)
        if start == -1:
            return (-1, -1)
        end = start + len(entity_text)
        return (start, end)
    
    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        标准化实体类型字符串。
        
        Args:
            entity_type: 原始实体类型字符串
            
        Returns:
            str: 标准化后的实体类型
        """
        # 转大写并替换空格为下划线
        normalized = entity_type.upper().replace(" ", "_")
        return normalized
    
    def _validate_entity(
        self, 
        entity: Dict[str, Any], 
        allowed_entities: Optional[List[str]] = None
    ) -> bool:
        """
        验证实体信息是否有效。
        
        Args:
            entity: 实体字典
            allowed_entities: 允许的实体类型列表
            
        Returns:
            bool: 实体是否有效
        """
        # 检查必需的字段
        if not all(k in entity for k in ["entity_type", "start", "end"]):
            return False
            
        # 检查位置是否有效
        if entity["start"] < 0 or entity["end"] <= entity["start"]:
            return False
            
        # 检查实体类型是否在允许列表中
        if allowed_entities and entity["entity_type"] not in allowed_entities:
            return False
            
        return True
