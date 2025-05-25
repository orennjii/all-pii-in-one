"""
文本处理模块，提供文本PII敏感信息识别与匿名化功能。

此模块包含:
1. 基于模式匹配和大型语言模型的各种识别器实现
2. 文本分割工具
3. 文本匿名化引擎
4. 完整的文本处理流程

用于识别、提取和匿名化中文文本中的敏感个人信息。
"""

# 核心模块
from .core import (
    PresidioAnalyzer,
    PresidioAnonymizer,
    TextSegmenter,
    TextSegment
)

# 主处理器
from .processor import TextProcessor, ProcessingResult

# 注册函数和识别器
from .utils.registry import register_chinese_pattern_recognizers, get_all_chinese_pattern_recognizers
from .recognizers.pattern import (
    BankCardRecognizer,
    CarPlateNumberRecognizer,
    IDCardRecognizer,
    PhoneNumberRecognizer,
    URLRecognizer,
)
from .recognizers.llm import LLMRecognizer

__all__ = [
    # 核心模块
    "PresidioAnalyzer",
    "PresidioAnonymizer", 
    "TextSegmenter",
    "TextSegment",
    
    # 主处理器
    "TextProcessor", "ProcessingResult",
    
    # 注册函数
    "register_chinese_pattern_recognizers",
    "get_all_chinese_pattern_recognizers",
    
    # 模式识别器
    "BankCardRecognizer",
    "CarPlateNumberRecognizer", 
    "IDCardRecognizer",
    "PhoneNumberRecognizer",
    "URLRecognizer",
    
    # LLM识别器
    "LLMRecognizer",
]
