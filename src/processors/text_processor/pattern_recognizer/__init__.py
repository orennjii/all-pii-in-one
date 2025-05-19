from .prc_cpn_recognizer import ChineseCarPlateNumberRecognizer
from .prc_pn_recognizer import ChineseMobileNumberRecognizer
from .prc_idc_recognizer import ChineseIDCardRecognizer
from .prc_cc_recognizer import ChineseBankCardRecognizer
from .prc_utl_recognizer import WebsiteURLRecognizer
from presidio_analyzer import RecognizerRegistry

# 导出所有自定义识别器
__all__ = [
    "ChineseCarPlateNumberRecognizer",
    "ChineseMobileNumberRecognizer", 
    "ChineseIDCardRecognizer",
    "ChineseBankCardRecognizer",
    "WebsiteURLRecognizer",
    "get_all_chinese_pattern_recognizers",
    "register_chinese_pattern_recognizers"
]

def get_all_chinese_pattern_recognizers():
    """返回所有中文自定义识别器的实例列表"""
    return [
        ChineseIDCardRecognizer(),
        ChineseCarPlateNumberRecognizer(),
        ChineseMobileNumberRecognizer(),
        ChineseBankCardRecognizer(),
        WebsiteURLRecognizer()
    ]

def register_chinese_pattern_recognizers(registry):
    """
    向RecognizerRegistry注册所有中文自定义识别器
    
    Args:
        registry (RecognizerRegistry): Presidio分析器的识别器注册表
    
    Returns:
        RecognizerRegistry: 更新后的注册表
    """
    if not isinstance(registry, RecognizerRegistry):
        raise TypeError("registry 必须是 RecognizerRegistry 类型")
    
    # 获取所有识别器实例
    recognizers = get_all_chinese_pattern_recognizers()
    
    # 将每个识别器添加到注册表
    for recognizer in recognizers:
        registry.add_recognizer(recognizer)
    
    return registry