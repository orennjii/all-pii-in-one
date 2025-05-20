"""
车牌号识别器模块。
"""
from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
from typing import Optional, List, Tuple


class CarPlateNumberRecognizer(PatternRecognizer):
    """
    识别中国车牌号的模式识别器。
    
    支持以下格式:
    - 普通汽车车牌: 省份简称(1) + 字母(1) + 字母/数字(5)
    - 新能源汽车车牌: 省份简称(1) + 字母(1) + 字母/数字(6)
    - 特殊车牌: 军队、武警等
    - 使馆车牌
    - 港澳车牌
    """

    # 定义正则表达式模式
    PATTERNS = [
        Pattern(
            name="car_plate_pattern",
            # 普通汽车车牌: 省份简称(1) + 字母(1) + 字母/数字(5)
            regex=r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]"
                  r"{1}[A-Z]{1}[A-Z0-9]{5}",
            score=0.85
        ),
        Pattern(
            name="new_energy_car_plate",
            # 新能源汽车车牌: 省份简称(1) + 字母(1) + 字母/数字(6)
            regex=r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领]"
                  r"{1}[A-Z]{1}[A-Z0-9]{6}",
            score=0.85
        ),
        Pattern(
            name="special_car_plate",
            # 特殊车牌: 如军队、武警等
            regex=r"[WJQGB][A-Z0-9]{5}[A-Z0-9]?",
            score=0.7
        ),
        Pattern(
            name="embassy_car_plate",
            # 使馆车牌
            regex=r"使[A-Z0-9]{5}[A-Z0-9]?",
            score=0.85
        ),
        Pattern(
            name="hk_macao_car_plate",
            # 港澳车牌
            regex=r"粤Z[A-Z0-9]{5}",
            score=0.85
        )
    ]

    CONTEXT = [
        "车牌", "牌照", "车辆", "汽车", "机动车", "车牌号码", "行驶证", 
        "挂牌", "车辆识别", "交管", "交警", "违章", "违停", "停车",
        "车主", "驾驶", "驾照", "登记", "上牌", "车辆管理", "备案"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "zh",
        supported_entity: str = "CAR_PLATE",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        self.replacement_pairs = replacement_pairs if replacement_pairs else [
            ("·", ""), ("-", ""), ("—", ""), (" ", ""), ("　", "")
        ]
        
        # 调用父类构造函数
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language
        )
    
    def validate_result(self, pattern_text: str) -> bool:
        """
        验证匹配的车牌号是否符合中国车牌号规则。
        
        Args:
            pattern_text: 待验证的文本
            
        Returns:
            bool: 验证通过返回True，否则返回False
        """
        # 清理文本，移除特殊字符
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        )
        
        # 检查长度：普通车牌7位，新能源车牌8位
        if len(sanitized_value) not in [7, 8]:
            return False
        
        # 检查省份缩写
        provinces = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
        first_char = sanitized_value[0]
        
        # 处理普通省份车牌
        if first_char in provinces:
            # 第二位必须是字母
            if not sanitized_value[1].isalpha() or not sanitized_value[1].isupper():
                return False
                
            # 后5位或6位可以是字母或数字，但至少有一个是数字(普通车牌规则)
            if len(sanitized_value) == 7 and not any(c.isdigit() for c in sanitized_value[2:]):
                return False
                
            # 新能源车牌通常以D或F结尾，且必须有数字
            if len(sanitized_value) == 8:
                if not (sanitized_value[-1] in "DF" or sanitized_value[-1].isdigit()):
                    return False
                if not any(c.isdigit() for c in sanitized_value[2:]):
                    return False
            
            return True
            
        # 处理特殊车牌（军队、武警等）
        elif first_char in "WJQGB":
            # 检查长度和格式
            return len(sanitized_value) in [6, 7] and all(c.isalnum() 
                                                          for c in sanitized_value[1:])
        
        return False
