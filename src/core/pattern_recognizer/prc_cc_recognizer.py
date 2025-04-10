from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
from typing import Optional, List, Tuple

class ChineseBankCardRecognizer(PatternRecognizer):
    """
    识别中国银行卡号的模式识别器
    
    支持以下格式:
    - 借记卡: 通常16-19位数字
    - 信用卡: 通常16位数字
    - 各种分隔符格式: 空格、短横线、点号等分隔的卡号
    
    主要校验:
    - 长度验证: 16-19位数字
    - Luhn算法校验: 校验码验证
    - BIN前缀校验: 银行卡发卡行识别码
    """

    # 定义正则表达式模式
    PATTERNS = [
        Pattern(
            name="bank_card_standard",
            # 标准银行卡号: 16-19位连续数字
            regex=r"\d{16,19}",
            score=0.7
        ),
        Pattern(
            name="bank_card_with_separator",
            # 带分隔符的银行卡号: 4位一组，用空格、短横线或点号分隔
            regex=r"\d{4}[\s\-\.]{1}\d{4}[\s\-\.]{1}\d{4}[\s\-\.]{1}\d{4}([\s\-\.]{1}\d{0,3})?",
            score=0.75
        ),
        Pattern(
            name="bank_card_standard_visa",
            # VISA卡: 以4开头，后跟15位数字
            regex=r"4\d{15}",
            score=0.8
        ),
        Pattern(
            name="bank_card_standard_mastercard",
            # MasterCard: 以5开头，后跟15位数字
            regex=r"5[1-5]\d{14}",
            score=0.8
        ),
        Pattern(
            name="bank_card_standard_unionpay",
            # 银联卡: 以62开头，后跟14-17位数字
            regex=r"62\d{14,17}",
            score=0.85
        ),
        Pattern(
            name="bank_card_with_bin",
            # 特定BIN前缀的银行卡
            regex=r"(62|35|37|4|5[1-5]|36|30[0-5])\d{14,17}",
            score=0.75
        )
    ]

    CONTEXT = [
        "银行卡", "卡号", "银行", "储蓄卡", "信用卡", "借记卡", "贷记卡", "信用额度",
        "中国银行", "工商银行", "农业银行", "建设银行", "交通银行", "招商银行", "邮储银行",
        "浦发银行", "民生银行", "兴业银行", "广发银行", "光大银行", "平安银行", "华夏银行",
        "银联", "支付", "转账", "汇款", "收款", "付款", "还款", "提现", "存款", "余额",
        "卡片", "取款", "账户", "账号", "银行账户", "银行账号", "金融", "理财",
        "Visa", "MasterCard", "UnionPay", "银联卡", "信用", "消费", "刷卡"
    ]

    # 主要银行卡BIN前缀
    BANK_BINS = {
        # 工商银行
        '620200': '工商银行', '620302': '工商银行', 
        # 农业银行
        '622848': '农业银行', '622700': '农业银行',
        # 中国银行
        '621660': '中国银行', '621661': '中国银行',
        # 建设银行
        '621700': '建设银行', '622280': '建设银行',
        # 交通银行
        '622260': '交通银行', '622262': '交通银行',
        # 邮储银行
        '621799': '邮储银行', '621095': '邮储银行',
        # 招商银行
        '621286': '招商银行', '622580': '招商银行',
        # 其他主要银行前6位BIN码...
    }

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "zh",
        supported_entity: str = "BANK_CARD",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        self.replacement_pairs = replacement_pairs if replacement_pairs else [
            ("-", ""), (" ", ""), ("　", ""), (".", ""), ("、", ""),
            ("卡号", ""), ("：", ""), (":", ""), ("号码", ""), ("银行卡", "")
        ]
        
        # 调用父类构造函数
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language
        )
    
    def validate_result(self, pattern_text: str) -> bool:
        """验证匹配的银行卡号是否符合规则"""
        # 清理文本，移除特殊字符
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        )
        
        # 检查是否全是数字
        if not sanitized_value.isdigit():
            return False
        
        # 检查长度是否在合理范围内(16-19位)
        length = len(sanitized_value)
        if length < 16 or length > 19:
            return False
        
        # 验证Luhn算法校验码
        if not self._check_luhn_algorithm(sanitized_value):
            return False
        
        # 验证BIN前缀(可选，不是所有卡都能匹配到，因此不作为强制条件)
        # self._check_bin_prefix(sanitized_value)
        
        return True
    
    def _check_luhn_algorithm(self, card_number: str) -> bool:
        """
        使用Luhn算法验证银行卡号
        
        算法步骤:
        1. 从右到左对每位数字进行处理
        2. 对奇数位(从右向左计数)直接取值
        3. 对偶数位数字乘以2，如果结果大于9，则减去9
        4. 将所有处理后的数字相加
        5. 如果总和能被10整除，则验证通过
        """
        digits = [int(digit) for digit in card_number]
        checksum = 0
        
        # 从右向左处理
        for i in range(len(digits) - 1, -1, -1):
            digit = digits[i]
            
            # 奇数位(从右数第1、3、5...位)
            if (len(digits) - i) % 2 == 1:
                checksum += digit
            # 偶数位(从右数第2、4、6...位)
            else:
                doubled = digit * 2
                # 如果乘以2后结果大于9，则减去9
                checksum += doubled if doubled < 10 else doubled - 9
        
        # 验证总和是否能被10整除
        return checksum % 10 == 0
    
    def _check_bin_prefix(self, card_number: str) -> bool:
        """验证银行卡BIN前缀是否符合已知银行发卡规则"""
        # 检查前6位是否在BIN列表中
        bin_6 = card_number[:6]
        bin_4 = card_number[:4]
        
        # 检查是否匹配已知BIN
        if bin_6 in self.BANK_BINS:
            return True
        
        # 检查一些通用规则
        # 银联卡以62开头
        if card_number.startswith('62'):
            return True
        # Visa卡以4开头
        if card_number.startswith('4'):
            return True
        # MasterCard以51-55开头
        if card_number.startswith(('51', '52', '53', '54', '55')):
            return True
        # American Express以34或37开头
        if card_number.startswith(('34', '37')):
            return True
        # JCB卡以35开头
        if card_number.startswith('35'):
            return True
        # Discover卡以6011或65开头
        if card_number.startswith(('6011', '65')):
            return True
        
        # 如果没有匹配到任何规则，但通过了Luhn算法，也认为是有效的
        return False