"""
基于模式匹配的识别器模块，用于识别结构化的个人敏感信息。
"""

from .bank_card import BankCardRecognizer
from .car_plate import CarPlateNumberRecognizer
from .id_card import IDCardRecognizer
from .phone_number import PhoneNumberRecognizer
from .url import URLRecognizer

from presidio_analyzer import RecognizerRegistry

def get_all_chinese_pattern_recognizers() -> list:
    """
    返回所有中文自定义识别器的实例列表。
    
    Returns:
        List[EntityRecognizer]: 识别器实例列表
    """
    return [
        IDCardRecognizer(),
        CarPlateNumberRecognizer(),
        PhoneNumberRecognizer(),
        BankCardRecognizer(),
        URLRecognizer()
    ]

def register_chinese_pattern_recognizers(registry: RecognizerRegistry) -> RecognizerRegistry:
    """
    向RecognizerRegistry注册所有中文自定义识别器。
    
    Args:
        registry (RecognizerRegistry): Presidio分析器的识别器注册表
    
    Returns:
        RecognizerRegistry: 更新后的注册表
    
    Raises:
        TypeError: 如果registry不是RecognizerRegistry类型
    """
    if not isinstance(registry, RecognizerRegistry):
        raise TypeError("registry 必须是 RecognizerRegistry 类型")
    
    # 获取所有识别器实例
    recognizers = get_all_chinese_pattern_recognizers()
    
    # 将每个识别器添加到注册表
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)
    
    return registry

__all__ = [
    "BankCardRecognizer",
    "CarPlateNumberRecognizer", 
    "IDCardRecognizer",
    "PhoneNumberRecognizer",
    "URLRecognizer",
]
