#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器基础模块

该模块提供用于解析LLM响应的基础接口和实现。
所有具体的LLM解析器应该继承ResponseParser类并实现必要的方法。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
import re
import json

from presidio_analyzer import RecognizerResult

from src.commons.loggers import get_module_logger

logger = get_module_logger(__name__)


class ResponseParser(ABC):
    """
    LLM响应解析器基类
    
    定义了将LLM输出解析为标准格式的接口，用于提取识别的实体及其在原始文本中的位置。
    所有具体的解析器实现都应该继承这个类并实现指定的方法。
    """

    def __init__(self, **kwargs):
        """
        初始化解析器。
        
        Args:
            **kwargs: 解析器特定的参数
        """
        self.kwargs = kwargs
        # 默认置信度分数
        self.default_score = kwargs.get("default_score", 0.8)
        # 是否在JSON解析失败时尝试文本解析
        self.fallback_to_text = kwargs.get("fallback_to_text", True)
    
    @abstractmethod
    def parse(
        self, 
        llm_response: str,
        original_text: str, 
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析LLM响应并返回识别结果列表。
        
        Args:
            llm_response: LLM返回的原始文本响应
            original_text: 原始输入文本
            entities: 要查找的实体类型列表（可选）
            
        Returns:
            List[Dict[str, Any]]: 包含实体识别结果的列表，每个结果应该包含：
                - entity_type: 实体类型
                - start: 在原始文本中的起始位置
                - end: 在原始文本中的结束位置
                - confidence: 置信度分数
                - value: 实体值
                
        Raises:
            ValueError: 如果无法解析LLM响应
        """
        pass
    
    def _find_text_position(self, original_text: str, entity_text: str) -> Tuple[int, int]:
        """
        在原始文本中查找实体文本的位置。
        
        Args:
            original_text: 原始输入文本
            entity_text: 要查找的实体文本
            
        Returns:
            Tuple[int, int]: (start, end) 表示实体在原始文本中的起始和结束位置
        """
        if not entity_text or not original_text:
            return -1, -1
            
        # 简单的字符串查找可能会有问题，因为相同的实体可能出现多次
        # 这里只返回第一次出现的位置
        start = original_text.find(entity_text)
        if start == -1:
            # 尝试忽略空白字符进行模糊匹配
            entity_no_space = re.sub(r'\s+', '', entity_text)
            for i in range(len(original_text)):
                text_slice = original_text[i:i+len(entity_no_space)+10]  # 额外检查几个字符
                text_slice_no_space = re.sub(r'\s+', '', text_slice)
                if text_slice_no_space.startswith(entity_no_space):
                    end = i + len(text_slice_no_space[:len(entity_no_space)])
                    return i, end
            return -1, -1
        else:
            end = start + len(entity_text)
            return start, end
    
    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """
        从文本中提取JSON对象。
        
        Args:
            text: 包含JSON的文本
            
        Returns:
            Optional[Dict[str, Any]]: 解析出的JSON对象，如果解析失败则返回None
        """
        # 尝试找到JSON数组 [...] 或对象 {...}
        json_pattern = r'(\[.*\]|\{.*\})'
        json_match = re.search(json_pattern, text, re.DOTALL)
        
        if json_match:
            json_str = json_match.group(0)
            try:
                return json.loads(json_str)
            except json.JSONDecodeError:
                # 简单清理，移除一些常见的干扰字符
                cleaned = re.sub(r'```json|```', '', json_str).strip()
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    logger.warning(f"JSON解析失败，尝试进行文本解析")
                    return None
        
        return None
        
    def _normalize_entity_type(self, entity_type: str) -> str:
        """
        规范化实体类型名称。
        
        Args:
            entity_type: 原始实体类型
            
        Returns:
            str: 规范化后的实体类型
        """
        if not entity_type:
            return ""
            
        # 转换为大写并替换空格为下划线
        normalized = entity_type.upper().replace(' ', '_')
        
        # 映射常见别名到标准名称
        mapping = {
            "PERSON_NAME": "PERSON",
            "NAME": "PERSON",
            "PERSONAL_NAME": "PERSON",
            "PHONE": "PHONE_NUMBER",
            "MOBILE": "PHONE_NUMBER",
            "MOBILE_PHONE": "PHONE_NUMBER",
            "EMAIL": "EMAIL_ADDRESS",
            "MAIL": "EMAIL_ADDRESS",
            "ID": "ID_CARD",
            "IDCARD": "ID_CARD",
            "CREDIT": "CREDIT_CARD",
            "CREDITCARD": "CREDIT_CARD",
            "BANK": "BANK_ACCOUNT",
            "BANKACCOUNT": "BANK_ACCOUNT",
            "LOCATION": "LOCATION",
            "ADDRESS": "ADDRESS",
            "IP": "IP_ADDRESS",
            "IPADDRESS": "IP_ADDRESS",
            "URL": "URL",
            "WEBSITE": "URL",
        }
        
        return mapping.get(normalized, normalized)