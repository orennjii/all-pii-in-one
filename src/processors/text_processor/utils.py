"""
文本处理工具函数。
"""

import re
from typing import List, Dict, Tuple, Optional


def clean_text(text: str, replacement_pairs: List[Tuple[str, str]] = None) -> str:
    """
    清理文本，移除或替换特殊字符。
    
    Args:
        text: 要清理的文本
        replacement_pairs: 要替换的字符对列表 [(旧字符, 新字符), ...]
        
    Returns:
        str: 清理后的文本
    """
    if not text:
        return ""
        
    if replacement_pairs:
        for old, new in replacement_pairs:
            text = text.replace(old, new)
            
    return text


def is_valid_pattern(text: str, pattern: str) -> bool:
    """
    检查文本是否匹配指定的正则表达式模式。
    
    Args:
        text: 要检查的文本
        pattern: 正则表达式模式
        
    Returns:
        bool: 匹配成功返回True，否则返回False
    """
    try:
        return bool(re.match(pattern, text))
    except Exception:
        return False
