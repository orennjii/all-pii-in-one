"""
文本匿名化处理器
集成文本分割、PII检测和匿名化功能
"""

import os
import re
import json
from typing import Dict, List, Tuple, Optional, Union, Any

from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

from .registry import register_chinese_pattern_recognizers
from .text_pii_detector import TextPIIDetector, PresidioPIIDetector, PatternPIIDetector
from .text_segmenter import TextSegmenter
from .text_anonymizer_engine import TextAnonymizerEngine


class TextProcessor:
    """
    文本处理器, 集成文本分割、PII检测和匿名化功能
    
    功能: 
    1. 文本分割: 将文本分割为句子或段落
    2. PII检测: 检测文本中的个人隐私信息
    3. 文本匿名化: 将检测到的PII信息进行匿名化处理
    
    处理流程: 
    1. 输入文本 -> 文本分割 -> 文本片段
    2. 每个文本片段 -> PII检测 -> 检测PII位置和类型
    3. 每个文本片段 -> 文本匿名化 -> 匿名化PII
    4. 合并所有匿名化片段 -> 输出匿名化文本
    """

    def __init__(
        self,
        pii_detector: Optional[TextPIIDetector] = None,
        anonymizer_engine: Optional[TextAnonymizerEngine] = None,
        segmenter: Optional[TextSegmenter] = None,
        enable_segmentation: bool = False,
        **kwargs
    ):
        """
        初始化文本匿名化器
        
        Args:
            pii_detector: PII检测器,默认使用Presidio和PatternPIIDetector的组合
            anonymizer_engine: 匿名化引擎,默认使用TextAnonymizerEngine
            segmenter: 文本分割器,默认使用TextSegmenter
            enable_segmentation: 是否启用文本分割
            **kwargs: 额外参数
        """
        # 初始化PII检测器
        self.pii_detector = pii_detector or self._create_default_pii_detector()
        
        # 初始化匿名化引擎
        self.anonymizer_engine = anonymizer_engine or TextAnonymizerEngine()
        
        # 初始化文本分割器
        self.segmenter = segmenter or TextSegmenter()
        self.enable_segmentation = enable_segmentation
        
    def _create_default_pii_detector(self) -> TextPIIDetector:
        """
        创建默认的PII检测器
        
        Returns:
            TextPIIDetector: 默认PII检测器
        """
        # 使用Presidio的分析器引擎
        registry = RecognizerRegistry()
        register_chinese_pattern_recognizers(registry)  # 注册中文识别器
        presidio_analyzer = AnalyzerEngine(registry=registry)
        
        # 创建Presidio检测器
        presidio_detector = PresidioPIIDetector(analyzer=presidio_analyzer)
        
        # 创建基于模式的检测器
        pattern_detector = PatternPIIDetector()
        
        # 组合多个检测器
        return TextPIIDetector(detectors=[presidio_detector, pattern_detector])
        
    def anonymize(
        self, 
        text: str, 
        language: str = 'zh',
        anonymization_strategy: Dict = None,
        segment_by: str = 'sentence',
        return_pii_info: bool = False,
        **kwargs
    ) -> Union[str, Tuple[str, Dict]]:
        """
        匿名化文本
        
        Args:
            text: 要匿名化的文本
            language: 文本语言
            anonymization_strategy: 匿名化策略,如{'PERSON': 'replace', 'PHONE_NUMBER': 'mask'}
            segment_by: 文本分割方式,'sentence'或'paragraph'
            return_pii_info: 是否返回PII信息
            **kwargs: 额外参数
            
        Returns:
            str或Tuple[str, Dict]: 匿名化后的文本,如果return_pii_info为True,还会返回PII信息
        """
        # 1. 文本分割
        if self.enable_segmentation:
            segments = self.segmenter.segment(text, by=segment_by)
        else:
            segments = [text]
            
        # 2. PII检测
        all_pii_results = []
        for i, segment in enumerate(segments):
            pii_results = self.pii_detector.detect(segment, language=language)
            # 更新PII位置使其相对于整个文本的偏移
            self._update_pii_offsets(pii_results, i, segments)
            all_pii_results.extend(pii_results)
            
        # 3. 文本匿名化
        anonymized_text, pii_mapping = self.anonymizer_engine.anonymize(
            text=text, 
            pii_results=all_pii_results, 
            strategy=anonymization_strategy or {}
        )
        
        # 4. 返回结果
        if return_pii_info:
            return anonymized_text, {
                'pii_results': all_pii_results,
                'pii_mapping': pii_mapping
            }
        return anonymized_text
        
    def _update_pii_offsets(self, pii_results: List[Dict], segment_index: int, segments: List[str]):
        """
        更新PII偏移量,使其相对于整个文本
        
        Args:
            pii_results: PII检测结果
            segment_index: 当前片段索引
            segments: 所有文本片段
        """
        if segment_index == 0:
            return
        
        # 计算之前片段的总长度
        offset = sum(len(segments[i]) for i in range(segment_index))
        
        # 更新每个PII的开始和结束位置
        for result in pii_results:
            result['start'] += offset
            result['end'] += offset
    
    def restore(
        self, 
        anonymized_text: str, 
        pii_mapping: Dict[str, str], 
        **kwargs
    ) -> str:
        """
        恢复匿名化文本
        
        Args:
            anonymized_text: 匿名化后的文本
            pii_mapping: PII映射关系
            **kwargs: 额外参数
            
        Returns:
            str: 恢复后的文本
        """
        return self.anonymizer_engine.restore(anonymized_text, pii_mapping)
    
    def batch_anonymize(
        self, 
        texts: List[str], 
        **kwargs
    ) -> List[Union[str, Tuple[str, Dict]]]:
        """
        批量匿名化文本
        
        Args:
            texts: 文本列表
            **kwargs: 传递给anonymize方法的参数
            
        Returns:
            List: 匿名化后的文本列表
        """
        return [self.anonymize(text, **kwargs) for text in texts]
    
    def detect_only(
        self, 
        text: str, 
        language: str = 'zh'
    ) -> List[Dict]:
        """
        仅检测PII,不进行匿名化
        
        Args:
            text: 要检测的文本
            language: 文本语言
            
        Returns:
            List[Dict]: PII检测结果
        """
        if self.enable_segmentation:
            segments = self.segmenter.segment(text)
            all_pii_results = []
            for i, segment in enumerate(segments):
                pii_results = self.pii_detector.detect(segment, language=language)
                self._update_pii_offsets(pii_results, i, segments)
                all_pii_results.extend(pii_results)
            return all_pii_results
        else:
            return self.pii_detector.detect(text, language=language)