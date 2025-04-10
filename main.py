from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import (
    AnonymizerEngine,
)
from src.core.pattern_recognizer import register_chinese_pattern_recognizers, get_all_chinese_pattern_recognizers

def main():
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{
            "lang_code": "zh",
            "model_name": "zh_core_web_lg"
        },
        {
            "lang_code": "en",
            "model_name": "en_core_web_lg"
        }]
    }).create_engine()

    # 创建注册表并注册所有自定义识别器
    registry = RecognizerRegistry(supported_languages=["zh", "en"])
    # registry.load_predefined_recognizers(languages=["zh", "en"])
    registry.add_nlp_recognizer(nlp_engine=nlp_engine)
    registry = register_chinese_pattern_recognizers(registry)

    # 使用自定义注册表创建分析器引擎
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["zh", "en"]
    )
    anonymizer = AnonymizerEngine()

    print("已注册的识别器详情:")
    for recognizer in analyzer.registry.recognizers:
        print(f"- {recognizer.__class__.__name__}: 实体类型={recognizer.supported_entities}, 语言={recognizer.supported_language}")

    # 测试文本，包含中文敏感信息
    cn_text = """
    这是一份包含敏感信息的示例文本：

    我的姓名是张三，来自北京。
    我的车牌号是京A12345，手机号码是13812345678。
    我的身份证号是210102199203183096，银行卡号是4929717705917895。

    2023年10月15日，我访问了www.example.com并发送邮件到test@example.com，IP地址是192.168.1.1。
    """

    en_text = """
    Here are a few example sentences we currently support:

    Hi, my name is David Johnson and I'm originally from Liverpool.
    My credit card number is 4095-2609-9393-4932 and my crypto wallet id is 16Yeky6GMjeNkAiNcBY7ZhrLoMSgg1BoyZ.

    On 11/10/2024 I visited www.microsoft.com and sent an email to test@presidio.site,  from IP 192.168.0.1.

    My passport: 191280342 and my phone number: (212) 555-1234.

    This is a valid International Bank Account Number: IL150120690000003111111 . Can you please check the status on bank account 954567876544? 

    Kate's social security number is 078-05-1126.  Her driver license? it is 1234567A.
    """

    # 分析并匿名化文本
    analyzed_results = analyzer.analyze(text=cn_text, language="zh")
    print(analyzed_results)
    # analyzed_results_en = analyzer.analyze(text=en_text, language="en")
    # print(analyzed_results_en)
    anonymized_text = anonymizer.anonymize(text=cn_text, analyzer_results=analyzed_results)
    print(anonymized_text)

if __name__ == "__main__":
    main()