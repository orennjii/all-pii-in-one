from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
from typing import Optional, List, Tuple
import datetime

class ChineseIDCardRecognizer(PatternRecognizer):
    """
    识别中国大陆居民身份证号的模式识别器
    
    支持以下格式:
    - 18位二代身份证: XXXXXX YYYYMMDD SSSX
    - 15位一代身份证: XXXXXX YYMMDD SSS
    - 支持各种分隔符格式
    
    其中:
    - XXXXXX: 6位地区编码
    - YYYYMMDD/YYMMDD: 出生日期
    - SSS(X): 顺序码，最后一位可能是校验码X
    """

    # 定义正则表达式模式
    PATTERNS = [
        Pattern(
            name="id_card_18",
            # 18位身份证
            regex=r"[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}(?:\d|X|x)",
            score=1
        ),
        Pattern(
            name="id_card_15",
            # 15位身份证(老身份证)
            regex=r"[1-9]\d{5}\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}",
            score=0.7
        ),
        Pattern(
            name="id_card_with_separator",
            # 带分隔符的身份证号(空格、短横线等)
            regex=r"[1-9]\d{5}[\s\-](?:19|20)\d{2}[\s\-]?(?:0[1-9]|1[0-2])[\s\-]?(?:0[1-9]|[12]\d|3[01])[\s\-]?\d{3}[\s\-]?(?:\d|X|x)",
            score=0.75
        ),
        Pattern(
            name="id_card_15_with_separator",
            # 带分隔符的15位身份证号
            regex=r"[1-9]\d{5}[\s\-]?\d{2}[\s\-]?(?:0[1-9]|1[0-2])[\s\-]?(?:0[1-9]|[12]\d|3[01])[\s\-]?\d{3}",
            score=0.65
        )
    ]

    CONTEXT = [
        "身份证", "身份证号", "身份证号码", "居民身份证", "公民身份号码", "公民身份证号码",
        "证件号", "证件号码", "证件", "户口", "户籍", "户口本", "户口簿", "户籍证明",
        "出生日期", "出生地", "籍贯", "人口", "登记", "核验", "核对", "二代身份证", 
        "居民证", "实名", "实名认证", "认证", "验证", "公安部", "信息", "个人信息",
        "注册", "登记", "办理", "申请", "领取", "补办", "换证", "年检"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "zh",
        supported_entity: str = "ID_CARD",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        self.replacement_pairs = replacement_pairs if replacement_pairs else [
            ("-", ""), (" ", ""), ("　", ""), (".", ""), ("、", ""),
            ("号", ""), ("：", ""), (":", ""), ("号码", ""), ("证件号", "")
        ]
        
        # 准备地区编码字典
        self._init_area_codes()
        
        # 调用父类构造函数
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language
        )
    
    def _init_area_codes(self):
        """初始化地区编码验证数据"""
        # 这里只列出部分省级编码前缀，实际使用时可以导入完整的地区编码表
        self.area_code_prefixes = {
            '11': '北京', '12': '天津', '13': '河北', '14': '山西', '15': '内蒙古',
            '21': '辽宁', '22': '吉林', '23': '黑龙江', 
            '31': '上海', '32': '江苏', '33': '浙江', '34': '安徽', '35': '福建',
            '36': '江西', '37': '山东',
            '41': '河南', '42': '湖北', '43': '湖南', '44': '广东', '45': '广西',
            '46': '海南',
            '50': '重庆', '51': '四川', '52': '贵州', '53': '云南', '54': '西藏',
            '61': '陕西', '62': '甘肃', '63': '青海', '64': '宁夏', '65': '新疆',
            '71': '台湾', '81': '香港', '82': '澳门', '91': '国外'
        }
    
    def validate_result(self, pattern_text: str) -> bool:
        """验证匹配的身份证号是否符合中国身份证规则"""
        # 清理文本，移除特殊字符
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        ).upper()  # 转为大写，因为最后一位校验码可能是X
        
        # 判断是15位还是18位
        length = len(sanitized_value)
        if length not in [15, 18]:
            return False
        
        # 检查是否都是数字(18位最后一位可能是X)
        if length == 18:
            if not (sanitized_value[:-1].isdigit() and (sanitized_value[-1].isdigit() or sanitized_value[-1] == 'X')):
                return False
        else:  # 15位
            if not sanitized_value.isdigit():
                return False
        
        # 检查地区编码
        area_code = sanitized_value[:2]
        if area_code not in self.area_code_prefixes:
            return False
        
        # 检查出生日期
        try:
            if length == 18:
                birth_date_str = sanitized_value[6:14]
                birth_date = datetime.datetime.strptime(birth_date_str, "%Y%m%d").date()
            else:  # 15位
                birth_year_prefix = '19'  # 15位身份证出生年份默认19xx
                birth_date_str = birth_year_prefix + sanitized_value[6:12]
                birth_date = datetime.datetime.strptime(birth_date_str, "%Y%m%d").date()
            
            # 检查日期是否合法且不超过当前日期
            current_date = datetime.datetime.now().date()
            if birth_date > current_date:
                return False
                
            # 检查日期是否在合理范围内(1900年至今)
            min_date = datetime.date(1900, 1, 1)
            if birth_date < min_date:
                return False
        except ValueError:
            # 日期格式不正确
            return False
        
        # 如果是18位身份证，验证校验码
        if length == 18:
            return self._verify_check_digit(sanitized_value)
        
        return True
    
    def _verify_check_digit(self, id_card: str) -> bool:
        """验证18位身份证的校验码"""
        # 身份证号前17位权重因子
        weight_factor = [7, 9, 10, 5, 8, 4, 2, 1, 6, 3, 7, 9, 10, 5, 8, 4, 2]
        # 校验码对应值
        check_code_map = '10X98765432'
        
        # 计算加权和
        weighted_sum = sum(int(id_card[i]) * weight_factor[i] for i in range(17))
        
        # 计算校验码
        check_code = check_code_map[weighted_sum % 11]
        
        # 验证校验码是否匹配
        return id_card[17] == check_code