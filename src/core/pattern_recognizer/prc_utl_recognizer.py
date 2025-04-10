from presidio_analyzer import PatternRecognizer, Pattern, EntityRecognizer
from typing import Optional, List, Tuple
import re

class WebsiteURLRecognizer(PatternRecognizer):
    """
    识别网站URL的模式识别器
    
    支持以下格式:
    - 完整URL: http(s)://domain.com/path?query
    - 无协议URL: www.domain.com, domain.com
    - 带子域名: subdomain.domain.com
    - 各种顶级域名: .com, .cn, .org, .edu.cn等
    """

    # 定义正则表达式模式
    PATTERNS = [
        Pattern(
            name="url_with_protocol",
            # 带协议的完整URL
            regex=r"\b(?:https?|ftp):\/\/[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            score=0.85
        ),
        Pattern(
            name="url_with_www",
            # 以www开头但无协议的URL
            regex=r"www\.[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            score=0.8
        ),
        Pattern(
            name="simple_domain",
            # 简单域名格式
            regex=r"\b[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            score=0.7
        ),
        Pattern(
            name="ip_address_url",
            # IP地址形式的URL
            regex=r"\b(?:https?:\/\/)?(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)(?::\d{1,5})?\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            score=0.75
        ),
        Pattern(
            name="cjk_domain",
            # 包含中日韩文字的URL(中文域名)
            regex=r"\b(?:https?:\/\/)?[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3400-\u4dbf]+\.(?:[a-z]{2,}|[\u4e00-\u9fff]+)\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
            score=0.8
        )
    ]

    CONTEXT = [
        "网站", "网址", "链接", "域名", "主页", "官网", "官方网站", "网页",
        "站点", "portal", "home", "网站地址", "访问", "浏览", "打开", "点击",
        "登录", "登陆", "注册", "http", "https", "www", "url", "URI",
        "网络", "互联网", "在线", "online", "网上", "云", "cloud",
        "浏览器", "browser", "访问地址", "超链接", "超文本", "hyperlink"
    ]

    def __init__(
        self,
        patterns: Optional[List[Pattern]] = None,
        context: Optional[List[str]] = None,
        supported_language: str = "zh",
        supported_entity: str = "URL",
        replacement_pairs: Optional[List[Tuple[str, str]]] = None
    ):
        patterns = patterns if patterns else self.PATTERNS
        context = context if context else self.CONTEXT
        self.replacement_pairs = replacement_pairs if replacement_pairs else [
            ("<", ""), (">", ""), ("「", ""), ("」", ""), (""", ""), (""", ""),
            ("'", ""), ("'", ""), ("网址:", ""), ("网址：", ""), ("地址:", ""), ("地址：", "")
        ]
        
        # 调用父类构造函数
        super().__init__(
            supported_entity=supported_entity,
            patterns=patterns,
            context=context,
            supported_language=supported_language
        )
    
    def validate_result(self, pattern_text: str) -> bool:
        """验证匹配的URL是否符合基本规则"""
        # 清理文本，移除特殊字符
        sanitized_value = EntityRecognizer.sanitize_value(
            pattern_text, self.replacement_pairs
        )
        
        # URL不应为空
        if not sanitized_value:
            return False
        
        # 基本验证：确保URL至少包含一个点，形成域名结构
        if "." not in sanitized_value:
            return False
        
        # 检查URL是否包含有效的顶级域名
        tld_pattern = r'\.(com|net|org|gov|edu|io|ai|app|co|info|cn|com\.cn|org\.cn|edu\.cn|gov\.cn|biz|top|xyz|site|online|tech|dev|me|tv|cc)($|\/|\?|#)'
        if not re.search(tld_pattern, sanitized_value, re.IGNORECASE):
            # 如果没有匹配常见顶级域名，但格式正确也可以接受
            # 例如一些不太常见的国家顶级域名或新顶级域名
            domain_pattern = r'(https?:\/\/|www\.)?([a-zA-Z0-9][-a-zA-Z0-9]{0,62}\.)+[a-zA-Z]{2,}($|\/|\?|#)'
            if not re.search(domain_pattern, sanitized_value, re.IGNORECASE):
                return False
        
        return True