from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import (
    AnonymizerEngine,
)
from src.core.pattern_recognizer import register_chinese_pattern_recognizers

def main():
    nlp_engine = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [{
            "lang_code": "zh",
            "model_name": "zh_core_web_lg"
        }]
    }).create_engine()

    # 创建注册表并注册所有自定义识别器
    registry = RecognizerRegistry(supported_languages=["zh", "en"])
    # registry.load_predefined_recognizers(languages=["zh"])
    # registry.add_nlp_recognizer(nlp_engine=nlp_engine)
    registry = register_chinese_pattern_recognizers(registry)
    print(f"注册的识别器: {[recognizer.__class__.__name__ for recognizer in registry.recognizers]}")

    # 使用自定义注册表创建分析器引擎
    analyzer = AnalyzerEngine(
        registry=registry,
        nlp_engine=nlp_engine,
        supported_languages=["zh", "en"]
    )
    anonymizer = AnonymizerEngine()

    # 测试文本，包含中文敏感信息
    text = """
    这是一份包含敏感信息的示例文本：

    我的姓名是张三，来自北京。
    我的车牌号是京A12345，手机号码是13812345678。
    我的身份证号是110101199001011234，银行卡号是6222021234567890123。

    2023年10月15日，我访问了www.example.com并发送邮件到test@example.com，IP地址是192.168.1.1。
    """

    # 分析并匿名化文本
    analyzed_results = analyzer.analyze(text=text, language="zh")
    print(analyzed_results)
    anonymized_text = anonymizer.anonymize(text=text, analyzer_results=analyzed_results)
    print(anonymized_text)

if __name__ == "__main__":
    main()