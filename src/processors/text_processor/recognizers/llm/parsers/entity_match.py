#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实体匹配数据结构模块

定义用于表示LLM识别结果的数据结构，确保与presidio框架的兼容性。
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class EntityMatch:
    """
    表示识别到的实体匹配信息
    
    该类用于封装LLM识别到的实体信息，包括实体类型、位置、置信度等，
    便于后续转换为presidio的RecognizerResult格式。
    """
    
    entity_type: str
    """实体类型，如PERSON、ID_CARD、PHONE_NUMBER等"""
    
    value: str
    """实体的原始文本值"""
    
    start: int
    """实体在原文中的起始位置（字符索引）"""
    
    end: int
    """实体在原文中的结束位置（字符索引）"""
    
    confidence: float
    """识别置信度，范围[0.0, 1.0]"""
    
    is_inferred: Optional[bool] = None
    """是否为推断得出的实体（可选）"""
    
    notes: Optional[str] = None
    """额外的说明或推断依据（可选）"""
    
    metadata: Optional[Dict[str, Any]] = None
    """额外的元数据信息（可选）"""
    
    def __post_init__(self) -> None:
        """初始化后的验证"""
        if self.start < 0:
            raise ValueError("start position must be non-negative")
        if self.end <= self.start:
            raise ValueError("end position must be greater than start position")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be between 0.0 and 1.0")
        if not self.entity_type:
            raise ValueError("entity_type cannot be empty")
        if not self.value:
            raise ValueError("value cannot be empty")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式
        
        Returns:
            Dict[str, Any]: 字典表示的实体匹配信息
        """
        result = {
            "entity_type": self.entity_type,
            "value": self.value,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence
        }
        
        if self.is_inferred is not None:
            result["is_inferred"] = self.is_inferred
        
        if self.notes:
            result["notes"] = self.notes
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EntityMatch":
        """
        从字典创建EntityMatch实例
        
        Args:
            data: 包含实体信息的字典
            
        Returns:
            EntityMatch: 创建的实体匹配实例
            
        Raises:
            KeyError: 当缺少必需字段时
            ValueError: 当字段值无效时
        """
        try:
            return cls(
                entity_type=data["entity_type"],
                value=data["value"],
                start=data["start"],
                end=data["end"],
                confidence=data["confidence"],
                is_inferred=data.get("is_inferred"),
                notes=data.get("notes"),
                metadata=data.get("metadata")
            )
        except KeyError as e:
            raise KeyError(f"Missing required field: {e}") from e
    
    def __str__(self) -> str:
        """字符串表示"""
        return (f"EntityMatch(type={self.entity_type}, value='{self.value}', "
                f"start={self.start}, end={self.end}, confidence={self.confidence:.2f})")
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
