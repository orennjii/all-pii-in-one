"""
简单的测试脚本，用于测试重构后的text_processor模块。
"""

from src.processors.text_processor_new.registry import get_all_chinese_pattern_recognizers
from src.processors.text_processor_new.recognizers.pattern import (
    BankCardRecognizer,
    CarPlateNumberRecognizer,
    IDCardRecognizer,
    PhoneNumberRecognizer,
    URLRecognizer
)

def test_imports():
    """测试导入是否正常工作"""
    print("测试导入...")
    
    # 测试获取所有识别器
    recognizers = get_all_chinese_pattern_recognizers()
    assert len(recognizers) == 5, f"预期有5个识别器，但实际有{len(recognizers)}个"
    
    # 测试各个识别器类
    bank_card = BankCardRecognizer()
    assert bank_card.supported_entity == "BANK_CARD", f"预期实体类型为BANK_CARD，但实际为{bank_card.supported_entity}"
    
    car_plate = CarPlateNumberRecognizer()
    assert car_plate.supported_entity == "CAR_PLATE", f"预期实体类型为CAR_PLATE，但实际为{car_plate.supported_entity}"
    
    id_card = IDCardRecognizer()
    assert id_card.supported_entity == "ID_CARD", f"预期实体类型为ID_CARD，但实际为{id_card.supported_entity}"
    
    phone = PhoneNumberRecognizer()
    assert phone.supported_entity == "PHONE_NUMBER", f"预期实体类型为PHONE_NUMBER，但实际为{phone.supported_entity}"
    
    url = URLRecognizer()
    assert url.supported_entity == "URL", f"预期实体类型为URL，但实际为{url.supported_entity}"
    
    print("导入测试通过!")
    
    return recognizers

if __name__ == "__main__":
    recognizers = test_imports()
    print("\n已创建的识别器:")
    for rec in recognizers:
        print(f"- {rec.__class__.__name__}: {rec.supported_entity} ({rec.supported_language})")
