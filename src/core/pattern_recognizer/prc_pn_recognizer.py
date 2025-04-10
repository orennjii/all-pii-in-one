from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
from typing import Optional, List, Tuple

class ChineseMobileNumberRecognizer(PatternRecognizer):
    """
    识别中国大陆手机号的模式识别器
    
    支持以下格式:
    - 11位数字: 1XXXXXXXXXX
    - 带分隔符: 1XX-XXXX-XXXX, 1XX XXXX XXXX
    - 带国家代码: +86 1XXXXXXXXXX, 0086-1XXXXXXXXXX
    """

    # 定义正则表达式模式
    PATTERNS = [
        Pattern(
            name="mobile_standard",
            # 标准11位手机号
            regex=r"\b1[3-9]\d{9}\b",
            score=0.8
        ),
        Pattern(
            name="mobile_with_separator",
            # 带分隔符的手机号(空格、短横线或点号)
            regex=r"1[3-9][\d\s\-\.]{9,13}",
            score=0.7
        ),
        Pattern(
            name="mobile_with_country_code",
            # 带国家代码的手机号：+86或0086
            regex=r"(?:\+86|0086)[- ]?1[3-9][\d\s\-\.]{9,13}",
            score=0.75
        ),
        Pattern(
            name="mobile_in_parentheses",
            # 括号中的手机号：(+86)XXXXX
            regex=r"\(\+?86\)[- ]?1[3-9][\d\s\-\.]{9,13}",
            score=0.75
        ),
        Pattern(
            name="mobile_special_format",
            # 特殊格式：1XX-XXXX-XXXX
            regex=r"1[3-9]\d{2}[\s\-\.]?\d{4}[\s\-\.]?\d{4}",
            score=0.75
        )
    ]

    CONTEXT = [
        "手机", "电话", "联系方式", "联系电话", "手机号", "手机号码", "电话号码",
        "拨打", "短信", "通话", "来电", "回电", "拨号", "联系人", "联系",
        "打电话", "给我打", "通知", "发短信", "接听", "转接", "咨询", "客服",
        "热线", "服务", "注册", "验证码", "验证", "联系我", "请联系"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "zh",
        supported_entity: str = "PHONE_NUMBER",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        self.replacement_pairs = replacement_pairs if replacement_pairs else [
            ("-", ""), (".", ""), (" ", ""), ("　", ""), ("(", ""), (")", ""),
            ("+86", ""), ("0086", ""), ("086", ""), ("+", "")
        ]
        
        # 调用父类构造函数
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language
        )
    
    def validate_result(self, pattern_text: str) -> bool:
        """验证匹配的手机号是否符合中国大陆手机号规则"""
        # 清理文本，移除特殊字符
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        )
        
        # 如果清理后的长度不是11位，则不是有效手机号
        if len(sanitized_value) != 11:
            return False
        
        # 检查是否都是数字
        if not sanitized_value.isdigit():
            return False
        
        # 检查手机号第一位必须是1
        if sanitized_value[0] != '1':
            return False
        
        # 检查手机号第二位必须是3-9之间的数字
        if sanitized_value[1] not in '3456789':
            return False
        
        # 检查上下文，避免识别身份证号中的片段
        # 如果这个数字序列是更长数字序列的一部分（如身份证号），则不认为是手机号
        import re
        # 获取匹配文本的前后字符，检查是否是更长数字序列的一部分
        match = re.search(r'\d{12,}', pattern_text)
        if match and len(match.group(0)) > 11:
            print(f"疑似身份证片段，不作为手机号识别: {pattern_text}")
            return False
        
        # 运营商前缀验证 (可选，但可以进一步提高准确性)
        prefix = sanitized_value[:3]
        
        # 移动: 134-139, 147-148, 150-152, 157-159, 165, 172, 178, 182-184, 187-188, 198
        # 联通: 130-132, 145-146, 155-156, 166, 171, 175-176, 185-186
        # 电信: 133, 149, 153, 173-174, 177, 180-181, 189, 199
        # 广电: 192
        # 虚拟运营商: 170
        
        # 2023年主流号段
        valid_prefixes = {
            # 移动
            '134', '135', '136', '137', '138', '139', '147', '148', '150', '151', '152', 
            '157', '158', '159', '165', '172', '178', '182', '183', '184', '187', '188', '198',
            # 联通
            '130', '131', '132', '145', '146', '155', '156', '166', '171', '175', '176', '185', '186',
            # 电信
            '133', '149', '153', '173', '174', '177', '180', '181', '189', '199',
            # 广电
            '192',
            # 虚拟运营商
            '170', '171'
        }
        
        # 非严格校验，考虑到号段会不断更新，如果不在列表中也接受
        # 如需严格校验，可取消下面注释
        # if prefix not in valid_prefixes:
        #     return False
        
        return True