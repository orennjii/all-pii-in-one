
import unittest
from presidio_analyzer import (
    AnalyzerEngine,
    RecognizerRegistry,
    PatternRecognizer
)
from presidio_analyzer.nlp_engine import NlpEngineProvider
from src.core.pattern_recognizer import register_chinese_pattern_recognizers

class PatternRecognizerTest(unittest.TestCase):
    
    def setUp(self):
        # 设置NLP引擎
        self.nlp_engine = NlpEngineProvider(nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [{
                "lang_code": "zh",
                "model_name": "zh_core_web_lg"
            }]
        }).create_engine()
        
        # 创建注册表并注册自定义识别器
        self.registry = RecognizerRegistry()
        self.registry.load_predefined_recognizers()
        self.registry = register_chinese_pattern_recognizers(self.registry)
        
        # 创建分析器
        self.analyzer = AnalyzerEngine(
            registry=self.registry, 
            nlp_engine=self.nlp_engine
        )
        
        # 获取注册的识别器列表
        self.recognizer_names = [recognizer.__class__.__name__ 
                               for recognizer in self.registry.recognizers]
        print(f"注册的识别器: {self.recognizer_names}")

    def test_car_plate_recognizer(self):
        """测试车牌号识别器"""
        text = "我的车牌号是京A12345。"
        results = self.analyzer.analyze(
            text=text, 
            language="zh",
            score_threshold=0.3
        )
        print(f"\n车牌测试结果: {results}")
        
        # 检查是否找到了车牌号
        car_plate_results = [r for r in results if r.entity_type == "CAR_PLATE"]
        self.assertTrue(len(car_plate_results) > 0, "未能识别车牌号")
        
        # 验证提取的值是否正确
        if car_plate_results:
            plate_text = text[car_plate_results[0].start:car_plate_results[0].end]
            self.assertEqual(plate_text, "京A12345")

    def test_phone_number_recognizer(self):
        """测试手机号码识别器"""
        text = "我的手机号码是13812345678。"
        results = self.analyzer.analyze(
            text=text, 
            language="zh",
            score_threshold=0.3
        )
        print(f"\n手机号测试结果: {results}")
        
        # 检查是否找到了手机号
        phone_results = [r for r in results if r.entity_type == "PHONE_NUMBER"]
        self.assertTrue(len(phone_results) > 0, "未能识别手机号")
        
        # 验证提取的值是否正确
        if phone_results:
            phone_text = text[phone_results[0].start:phone_results[0].end]
            self.assertEqual(phone_text, "13812345678")
            
    def test_id_card_recognizer(self):
        """测试身份证号识别器"""
        text = "我的身份证号是110101199001011234。"
        results = self.analyzer.analyze(
            text=text, 
            language="zh",
            score_threshold=0.3
        )
        print(f"\n身份证测试结果: {results}")
        
        # 检查是否找到了身份证号
        id_results = [r for r in results if r.entity_type == "ID_CARD"]
        self.assertTrue(len(id_results) > 0, "未能识别身份证号")
        
        # 验证提取的值是否正确
        if id_results:
            id_text = text[id_results[0].start:id_results[0].end]
            self.assertEqual(id_text, "110101199001011234")
            
    def test_bank_card_recognizer(self):
        """测试银行卡号识别器"""
        text = "我的银行卡号是6222021234567890123。"
        results = self.analyzer.analyze(
            text=text, 
            language="zh",
            score_threshold=0.3
        )
        print(f"\n银行卡测试结果: {results}")
        
        # 检查是否找到了银行卡号
        card_results = [r for r in results if r.entity_type == "BANK_CARD"]
        self.assertTrue(len(card_results) > 0, "未能识别银行卡号")
        
        # 验证提取的值是否正确
        if card_results:
            card_text = text[card_results[0].start:card_results[0].end]
            self.assertEqual(card_text, "6222021234567890123")
    
    def test_all_recognizers_together(self):
        """测试所有识别器在一段文本中的效果"""
        text = """
        这是一份测试文档。
        我的姓名是张三，来自北京。
        我的车牌号是京A12345，手机号码是13812345678。
        我的身份证号是110101199001011234，银行卡号是6222021234567890123。
        """
        
        results = self.analyzer.analyze(
            text=text, 
            language="zh",
            score_threshold=0.3
        )
        
        print(f"\n综合测试结果: {results}")
        entity_types = [result.entity_type for result in results]
        
        # 检查是否找到了所有类型的敏感信息
        self.assertIn("CAR_PLATE", entity_types, "未能识别车牌号")
        self.assertIn("PHONE_NUMBER", entity_types, "未能识别手机号")
        self.assertIn("ID_CARD", entity_types, "未能识别身份证号")
        self.assertIn("BANK_CARD", entity_types, "未能识别银行卡号")

    def test_with_regex_directly(self):
        """直接使用正则表达式测试模式"""
        import re
        
        # 测试车牌号正则表达式
        car_plate_pattern = r"[京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领][A-Z][A-Z0-9]{5}"
        text = "我的车牌号是京A12345。"
        matches = re.findall(car_plate_pattern, text)
        print(f"\n直接正则测试-车牌: {matches}")
        self.assertTrue(len(matches) > 0, "正则表达式未能匹配车牌号")
        
        # 测试手机号正则表达式
        phone_pattern = r"1[3-9]\d{9}"
        text = "我的手机号码是13812345678。"
        matches = re.findall(phone_pattern, text)
        print(f"直接正则测试-手机号: {matches}")
        self.assertTrue(len(matches) > 0, "正则表达式未能匹配手机号")

if __name__ == "__main__":
    unittest.main()