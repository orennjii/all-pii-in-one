#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
TextProcessor类综合测试模块

该模块演示如何使用TextProcessor类进行完整的PII检测和匿名化处理。
包含以下测试场景：
1. 基础文本处理
2. 批量文本处理 
3. 仅分析模式
4. 仅匿名化模式
5. 仅分割模式
6. 自定义配置测试
7. 不同语言支持测试
8. 性能测试

可以通过以下方式运行:
python test/test_text_processor.py
或
PYTHONPATH=. python -m test.test_text_processor
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

# 添加项目根目录到Python路径
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent
sys.path.insert(0, str(project_root))

from src.commons.loggers import get_module_logger, init_logging
from src.commons.utils import find_project_root
from src.configs import AppConfig
from src.configs.processors.text_processor import TextProcessorConfig
from src.processors.text_processor import (
    TextProcessor, ProcessingResult, 
    create_text_processor, get_text_processor,
    TextProcessorFactory
)

logger = get_module_logger(__name__)


class TextProcessorTester:
    """TextProcessor测试类"""
    
    def __init__(self):
        """初始化测试器"""
        self.processor: Optional[TextProcessor] = None
        self.test_results: Dict[str, Any] = {}
        
        # 测试数据
        self.test_texts = {
            "simple": "张三的手机号是13912345678。",
            "complex": """
            张三是一位软件工程师，他的身份证号码是330102199901011234，
            电话号码是13800138000，邮箱是zhangsan@example.com。
            他的银行卡号是6222024000001234567，车牌号是浙A12345。
            他经常访问https://www.example.com网站进行学习。
            
            李四也是一名开发者，身份证号441881199912123456，
            联系电话15912345678，工作地址在北京市朝阳区建国门外大街1号。
            """,
            "chinese_only": "王五住在上海市浦东新区陆家嘴金融中心，身份证是310115198805156789。",
            "mixed": "Hello, 我是John Smith，电话：+86-138-0013-8000，邮箱：john@company.com。",
            "empty": "",
            "no_pii": "今天天气很好，适合出门散步。",
            "batch_test": [
                "第一个人：张三，电话13812345678",
                "第二个人：李四，身份证110101199001011234",
                "第三个人：王五，邮箱wangwu@example.com",
                "第四个人：赵六，银行卡6222020000000000000"
            ]
        }
    
    def setup_processor(self, config: Optional[TextProcessorConfig] = None) -> None:
        """设置处理器"""
        try:
            if config:
                self.processor = TextProcessor(config)
            else:
                # 使用默认配置或从配置文件加载
                config_path = find_project_root(current_path) / "config" / "app_config.yaml"
                if config_path.exists():
                    app_config = AppConfig.load_from_yaml(str(config_path))
                    self.processor = TextProcessor(app_config.text_processor)
                else:
                    # 使用工厂函数创建
                    self.processor = create_text_processor()
            
            logger.info("TextProcessor 初始化成功")
            
        except Exception as e:
            logger.error(f"TextProcessor 初始化失败: {e}")
            raise
    
    def test_basic_processing(self) -> Dict[str, Any]:
        """测试基础文本处理功能"""
        logger.info("开始测试基础文本处理功能...")
        
        if not self.processor:
            return {
                "processing_successful": False,
                "error": "处理器未初始化"
            }
        
        results = {}
        
        for name, text in self.test_texts.items():
            if name == "batch_test":
                continue  # 跳过批量测试数据
            
            try:
                logger.info(f"处理测试文本: {name}")
                
                result = self.processor.process(
                    text=text,
                    enable_segmentation=True,
                    enable_analysis=True,
                    enable_anonymization=True,
                    language="zh"
                )
                
                # 统计结果
                total_entities = sum(len(analysis) for analysis in result.analysis_results)
                
                results[name] = {
                    "original_length": len(text),
                    "segments_count": len(result.segments),
                    "entities_found": total_entities,
                    "anonymized_length": len(result.anonymized_text),
                    "processing_successful": True,
                    "anonymized_text": result.anonymized_text[:100] + "..." if len(result.anonymized_text) > 100 else result.anonymized_text
                }
                
                logger.info(f"  - 原文长度: {len(text)}")
                logger.info(f"  - 分段数量: {len(result.segments)}")
                logger.info(f"  - 检测到PII实体数: {total_entities}")
                logger.info(f"  - 匿名化后文本长度: {len(result.anonymized_text)}")
                
                if total_entities > 0:
                    logger.info("  - 检测到的实体:")
                    for i, analysis_results in enumerate(result.analysis_results):
                        for entity in analysis_results:
                            logger.info(f"    段落{i+1}: {entity.entity_type} - '{entity.text}' (置信度: {entity.score:.2f})")
                
            except Exception as e:
                logger.error(f"处理文本 '{name}' 时出错: {e}")
                results[name] = {
                    "processing_successful": False,
                    "error": str(e)
                }
        
        # 检查是否所有测试都成功
        all_successful = all(result.get("processing_successful", False) for result in results.values())
        
        # 添加总体成功标志
        results["_overall_success"] = all_successful
        
        return results
    
    def test_batch_processing(self) -> Dict[str, Any]:
        """测试批量处理功能"""
        logger.info("开始测试批量处理功能...")
        
        batch_texts = self.test_texts["batch_test"]
        
        try:
            start_time = time.time()
            
            results = self.processor.batch_process(
                texts=batch_texts,
                enable_segmentation=True,
                enable_analysis=True,
                enable_anonymization=True,
                language="zh"
            )
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # 统计批量处理结果
            total_entities = 0
            successful_count = 0
            
            for i, result in enumerate(results):
                if not result.metadata.get('error'):
                    successful_count += 1
                    total_entities += sum(len(analysis) for analysis in result.analysis_results)
                    
                    logger.info(f"批量处理 {i+1}: 检测到 {sum(len(analysis) for analysis in result.analysis_results)} 个实体")
                    logger.info(f"  原文: {batch_texts[i]}")
                    logger.info(f"  匿名化: {result.anonymized_text}")
                else:
                    logger.error(f"批量处理 {i+1} 失败: {result.metadata.get('error')}")
            
            batch_result = {
                "total_texts": len(batch_texts),
                "successful_count": successful_count,
                "total_entities": total_entities,
                "processing_time": processing_time,
                "average_time_per_text": processing_time / len(batch_texts),
                "processing_successful": True
            }
            
            logger.info(f"批量处理完成:")
            logger.info(f"  - 总文本数: {len(batch_texts)}")
            logger.info(f"  - 成功处理: {successful_count}")
            logger.info(f"  - 总实体数: {total_entities}")
            logger.info(f"  - 总耗时: {processing_time:.2f}秒")
            logger.info(f"  - 平均耗时: {processing_time / len(batch_texts):.2f}秒/文本")
            
            return batch_result
            
        except Exception as e:
            logger.error(f"批量处理失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_analysis_only(self) -> Dict[str, Any]:
        """测试仅分析功能"""
        logger.info("开始测试仅分析功能...")
        
        test_text = self.test_texts["complex"]
        
        try:
            analysis_results = self.processor.analyze_only(
                text=test_text,
                language="zh"
            )
            
            result = {
                "entities_found": len(analysis_results),
                "entities": [],
                "processing_successful": True
            }
            
            logger.info(f"仅分析模式检测到 {len(analysis_results)} 个实体:")
            for entity in analysis_results:
                entity_info = {
                    "type": entity.entity_type,
                    "text": entity.text,
                    "score": entity.score,
                    "start": entity.start,
                    "end": entity.end
                }
                result["entities"].append(entity_info)
                logger.info(f"  - {entity.entity_type}: '{entity.text}' (置信度: {entity.score:.2f}, 位置: {entity.start}-{entity.end})")
            
            return result
            
        except Exception as e:
            logger.error(f"仅分析测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_anonymization_only(self) -> Dict[str, Any]:
        """测试仅匿名化功能"""
        logger.info("开始测试仅匿名化功能...")
        
        test_text = "张三的电话是13812345678，邮箱是zhangsan@example.com"
        
        try:
            # 先进行分析
            analysis_results = self.processor.analyze_only(test_text, language="zh")
            
            # 然后进行匿名化
            anonymization_result = self.processor.anonymize_only(
                text=test_text,
                analyzer_results=analysis_results
            )
            
            result = {
                "original_text": test_text,
                "anonymized_text": anonymization_result.text,
                "anonymization_items": len(anonymization_result.items),
                "processing_successful": True
            }
            
            logger.info(f"仅匿名化测试:")
            logger.info(f"  原文: {test_text}")
            logger.info(f"  匿名化后: {anonymization_result.text}")
            logger.info(f"  匿名化项目数: {len(anonymization_result.items)}")
            
            return result
            
        except Exception as e:
            logger.error(f"仅匿名化测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_segmentation_only(self) -> Dict[str, Any]:
        """测试仅分割功能"""
        logger.info("开始测试仅分割功能...")
        
        test_text = "这是第一句话。这是第二句话！这是第三句话？还有第四句话。"
        
        try:
            segments = self.processor.segment_only(test_text)
            
            result = {
                "original_text": test_text,
                "segments_count": len(segments),
                "segments": [],
                "processing_successful": True
            }
            
            logger.info(f"仅分割测试:")
            logger.info(f"  原文: {test_text}")
            logger.info(f"  分割结果 ({len(segments)} 个段落):")
            
            for i, segment in enumerate(segments):
                segment_info = {
                    "index": i + 1,
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end
                }
                result["segments"].append(segment_info)
                logger.info(f"    段落{i+1}: '{segment.text}' (位置: {segment.start}-{segment.end})")
            
            return result
            
        except Exception as e:
            logger.error(f"仅分割测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_custom_configuration(self) -> Dict[str, Any]:
        """测试自定义配置"""
        logger.info("开始测试自定义配置...")
        
        try:
            # 创建自定义配置
            from src.configs.processors.text_processor.core import AnalyzerConfig, AnonymizerConfig, CoreConfig
            
            custom_config = TextProcessorConfig()
            
            # 设置自定义匿名化配置
            custom_config.core.anonymizer.entity_anonymization_config = {
                "PERSON": {
                    "operator": "replace",
                    "params": {"new_value": "[姓名]"}
                },
                "PHONE_NUMBER": {
                    "operator": "mask", 
                    "params": {"masking_char": "*", "chars_to_mask": 8, "from_end": True}
                },
                "ID_CARD": {
                    "operator": "replace",
                    "params": {"new_value": "[身份证]"}
                }
            }
            
            # 创建使用自定义配置的处理器
            custom_processor = TextProcessor(custom_config)
            
            test_text = "张三的身份证是330102199901011234，电话是13812345678"
            
            result = custom_processor.process(
                text=test_text,
                enable_segmentation=True,
                enable_analysis=True,
                enable_anonymization=True,
                language="zh"
            )
            
            custom_result = {
                "original_text": test_text,
                "anonymized_text": result.anonymized_text,
                "entities_found": sum(len(analysis) for analysis in result.analysis_results),
                "processing_successful": True,
                "config_test": "custom_anonymization"
            }
            
            logger.info(f"自定义配置测试:")
            logger.info(f"  原文: {test_text}")
            logger.info(f"  自定义匿名化后: {result.anonymized_text}")
            
            return custom_result
            
        except Exception as e:
            logger.error(f"自定义配置测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_processor_features(self) -> Dict[str, Any]:
        """测试处理器特性功能"""
        logger.info("开始测试处理器特性功能...")
        
        try:
            # 测试支持的实体类型
            supported_entities = self.processor.get_supported_entities()
            
            # 测试支持的操作符
            supported_operators = self.processor.get_supported_operators()
            
            features_result = {
                "supported_entities_count": len(supported_entities),
                "supported_entities": supported_entities,
                "supported_operators_count": len(supported_operators),
                "supported_operators": supported_operators,
                "analyzer_available": self.processor.analyzer is not None,
                "anonymizer_available": self.processor.anonymizer is not None,
                "segmenter_available": self.processor.segmenter is not None,
                "processing_successful": True
            }
            
            logger.info(f"处理器特性:")
            logger.info(f"  支持的实体类型 ({len(supported_entities)}个): {supported_entities}")
            logger.info(f"  支持的操作符 ({len(supported_operators)}个): {supported_operators}")
            logger.info(f"  分析器可用: {self.processor.analyzer is not None}")
            logger.info(f"  匿名化器可用: {self.processor.anonymizer is not None}")
            logger.info(f"  分割器可用: {self.processor.segmenter is not None}")
            
            return features_result
            
        except Exception as e:
            logger.error(f"处理器特性测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def test_performance(self) -> Dict[str, Any]:
        """测试性能"""
        logger.info("开始测试性能...")
        
        # 准备测试数据
        large_text = self.test_texts["complex"] * 10  # 重复10次增加文本量
        
        try:
            # 测试单次处理性能
            start_time = time.time()
            result = self.processor.process(
                text=large_text,
                enable_segmentation=True,
                enable_analysis=True,
                enable_anonymization=True,
                language="zh"
            )
            end_time = time.time()
            
            single_processing_time = end_time - start_time
            
            # 测试多次处理性能
            iterations = 5
            start_time = time.time()
            
            for i in range(iterations):
                self.processor.process(
                    text=self.test_texts["simple"],
                    enable_segmentation=True,
                    enable_analysis=True,
                    enable_anonymization=True,
                    language="zh"
                )
            
            end_time = time.time()
            multi_processing_time = end_time - start_time
            
            performance_result = {
                "large_text_length": len(large_text),
                "large_text_processing_time": single_processing_time,
                "large_text_entities_found": sum(len(analysis) for analysis in result.analysis_results),
                "multi_iterations": iterations,
                "multi_processing_time": multi_processing_time,
                "average_time_per_iteration": multi_processing_time / iterations,
                "processing_successful": True
            }
            
            logger.info(f"性能测试结果:")
            logger.info(f"  大文本长度: {len(large_text)} 字符")
            logger.info(f"  大文本处理时间: {single_processing_time:.2f}秒")
            logger.info(f"  大文本检测实体数: {sum(len(analysis) for analysis in result.analysis_results)}")
            logger.info(f"  多次处理({iterations}次)总时间: {multi_processing_time:.2f}秒")
            logger.info(f"  平均每次处理时间: {multi_processing_time / iterations:.2f}秒")
            
            return performance_result
            
        except Exception as e:
            logger.error(f"性能测试失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """运行所有测试"""
        logger.info("=" * 80)
        logger.info("开始运行 TextProcessor 综合测试")
        logger.info("=" * 80)
        
        all_results = {}
        
        try:
            # 设置处理器
            self.setup_processor()
            
            # 运行各项测试
            test_functions = [
                ("basic_processing", self.test_basic_processing),
                ("batch_processing", self.test_batch_processing),
                ("analysis_only", self.test_analysis_only),
                ("anonymization_only", self.test_anonymization_only),
                ("segmentation_only", self.test_segmentation_only),
                ("custom_configuration", self.test_custom_configuration),
                ("processor_features", self.test_processor_features),
                ("performance", self.test_performance),
            ]
            
            for test_name, test_function in test_functions:
                logger.info(f"\n{'-' * 60}")
                logger.info(f"运行测试: {test_name}")
                logger.info(f"{'-' * 60}")
                
                try:
                    result = test_function()
                    all_results[test_name] = result
                    
                    if result.get("processing_successful", False):
                        logger.info(f"✅ 测试 {test_name} 通过")
                    else:
                        logger.error(f"❌ 测试 {test_name} 失败: {result.get('error', '未知错误')}")
                
                except Exception as e:
                    logger.error(f"❌ 测试 {test_name} 执行异常: {e}")
                    all_results[test_name] = {
                        "processing_successful": False,
                        "error": str(e)
                    }
            
            # 汇总结果
            successful_tests = sum(1 for result in all_results.values() if result.get("processing_successful", False))
            total_tests = len(all_results)
            
            logger.info(f"\n{'=' * 80}")
            logger.info(f"测试完成 - 通过: {successful_tests}/{total_tests}")
            logger.info(f"{'=' * 80}")
            
            # 详细结果汇总
            all_results["summary"] = {
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": total_tests - successful_tests,
                "success_rate": successful_tests / total_tests if total_tests > 0 else 0
            }
            
            return all_results
            
        except Exception as e:
            logger.error(f"测试运行失败: {e}")
            return {
                "processing_successful": False,
                "error": str(e)
            }


def main():
    """主函数"""
    # 初始化日志
    import logging
    init_logging(level=logging.INFO)
    
    # 创建测试器
    tester = TextProcessorTester()
    
    try:
        # 运行所有测试
        results = tester.run_all_tests()
        
        # 输出最终统计
        summary = results.get("summary", {})
        if summary:
            print(f"\n🎯 最终测试统计:")
            print(f"   总测试数: {summary['total_tests']}")
            print(f"   成功测试: {summary['successful_tests']}")
            print(f"   失败测试: {summary['failed_tests']}")
            print(f"   成功率: {summary['success_rate']:.1%}")
            
            if summary['success_rate'] >= 0.8:
                print(f"🎉 TextProcessor 工作正常！")
            else:
                print(f"⚠️  TextProcessor 可能存在问题，请检查失败的测试。")
        
        return results
        
    except Exception as e:
        logger.error(f"测试执行失败: {e}")
        print(f"❌ 测试执行失败: {e}")
        return None


if __name__ == "__main__":
    results = main()
    
    # 如果需要，可以将结果保存到文件
    if results:
        import json
        
        # 保存测试结果
        results_file = project_root / "logs" / "text_processor_test_results.json"
        results_file.parent.mkdir(exist_ok=True)
        
        # 序列化结果（去除不可序列化的对象）
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                serializable_results[key] = value
            else:
                serializable_results[key] = str(value)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        
        print(f"\n📄 测试结果已保存到: {results_file}")
