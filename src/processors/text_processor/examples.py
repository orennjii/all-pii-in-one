"""
文本匿名化处理器示例

展示如何使用TextAnonymizer进行文本PII检测与匿名化
"""

from src.processors.text_processor_new import (
    TextProcessor,
    TextPIIDetector,
    PatternPIIDetector,
    TextSegmenter,
    TextAnonymizerEngine
)


def basic_usage_example():
    """
    基本用法示例
    """
    # 初始化文本匿名化器
    anonymizer = TextProcessor()
    
    # 准备测试文本
    text = """
    张三的手机号是13912345678，他住在北京市海淀区中关村南大街5号，
    身份证号码为110101199001011234。
    他在中国银行的卡号是6222021234567890123，邮箱是zhangsan@example.com。
    """
    
    # 进行匿名化处理
    anonymized_text = anonymizer.anonymize(
        text=text, 
        language='zh'
    )
    
    print("原始文本:")
    print(text)
    print("\n匿名化后文本:")
    print(anonymized_text)
    
    # 获取更详细的PII信息
    anonymized_text, pii_info = anonymizer.anonymize(
        text=text, 
        language='zh',
        return_pii_info=True
    )
    
    print("\nPII检测结果:")
    for result in pii_info['pii_results']:
        print(f"类型: {result['entity_type']}, 值: {result['value']}, 位置: {result['start']}-{result['end']}")
    

def custom_strategy_example():
    """
    自定义匿名化策略示例
    """
    # 初始化文本匿名化器
    anonymizer = TextProcessor()
    
    # 准备测试文本
    text = """
    张三的手机号是13912345678，他住在北京市海淀区中关村南大街5号，
    身份证号码为110101199001011234。
    """
    
    # 自定义匿名化策略
    strategy = {
        'PERSON': 'replace',     # 替换为虚构人名
        'PHONE_NUMBER': 'mask',  # 掩码处理 (*****)
        'ADDRESS': 'redact',     # 编辑处理 [已编辑]
        'ID_CARD': 'hash'        # 哈希处理
    }
    
    # 进行匿名化处理
    anonymized_text = anonymizer.anonymize(
        text=text, 
        language='zh',
        anonymization_strategy=strategy
    )
    
    print("原始文本:")
    print(text)
    print("\n自定义策略匿名化后文本:")
    print(anonymized_text)


def batch_processing_example():
    """
    批量处理示例
    """
    # 初始化文本匿名化器
    anonymizer = TextProcessor()
    
    # 准备多个测试文本
    texts = [
        "张三的手机号是13912345678",
        "李四的邮箱是lisi@example.com",
        "王五的身份证号是110101199001011234"
    ]
    
    # 批量匿名化
    anonymized_texts = anonymizer.batch_anonymize(texts, language='zh')
    
    print("批量匿名化结果:")
    for i, (original, anonymized) in enumerate(zip(texts, anonymized_texts)):
        print(f"文本 {i+1}:")
        print(f"原始: {original}")
        print(f"匿名化: {anonymized}")
        print()


def segmentation_example():
    """
    文本分割示例
    """
    # 初始化带有分割功能的文本匿名化器
    anonymizer = TextProcessor(enable_segmentation=True)
    
    # 准备长文本
    text = """
    第一段: 张三的手机号是13912345678。他住在北京市海淀区中关村南大街5号。
    他的电子邮件是zhangsan@example.com。
    
    第二段: 李四今年35岁，他的身份证号码是110101198501011234。
    他在中国工商银行的卡号是6222021234567890321。
    
    第三段: 这段文本不包含任何个人隐私信息。
    """
    
    # 进行分段匿名化处理
    anonymized_text = anonymizer.anonymize(
        text=text, 
        language='zh',
        segment_by='paragraph'  # 按段落分割
    )
    
    print("原始长文本:")
    print(text)
    print("\n分段匿名化后的文本:")
    print(anonymized_text)


def detection_only_example():
    """
    仅检测不匿名化示例
    """
    # 初始化文本匿名化器
    anonymizer = TextProcessor()
    
    # 准备测试文本
    text = """
    张三的手机号是13912345678，他住在北京市海淀区中关村南大街5号，
    身份证号码为110101199001011234。
    """
    
    # 仅检测PII
    pii_results = anonymizer.detect_only(text, language='zh')
    
    print("仅检测PII结果:")
    for result in pii_results:
        print(f"类型: {result['entity_type']}, 值: {result['value']}, 位置: {result['start']}-{result['end']}")


def restore_example():
    """
    匿名化恢复示例
    """
    # 初始化文本匿名化器
    anonymizer = TextProcessor()
    
    # 准备测试文本
    text = "张三的手机号是13912345678"
    
    # 进行匿名化处理并获取PII映射
    anonymized_text, pii_info = anonymizer.anonymize(
        text=text, 
        language='zh',
        return_pii_info=True
    )
    
    print("原始文本:", text)
    print("匿名化后:", anonymized_text)
    
    # 恢复原始文本
    restored_text = anonymizer.restore(
        anonymized_text=anonymized_text,
        pii_mapping=pii_info['pii_mapping']
    )
    
    print("恢复后:", restored_text)
    print("恢复成功:", restored_text == text)


if __name__ == "__main__":
    print("===== 基本用法示例 =====")
    basic_usage_example()
    
    print("\n\n===== 自定义匿名化策略示例 =====")
    custom_strategy_example()
    
    print("\n\n===== 批量处理示例 =====")
    batch_processing_example()
    
    print("\n\n===== 文本分割示例 =====")
    segmentation_example()
    
    print("\n\n===== 仅检测不匿名化示例 =====")
    detection_only_example()
    
    print("\n\n===== 匿名化恢复示例 =====")
    restore_example()
