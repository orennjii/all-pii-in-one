#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理器主模块

提供统一的文本PII处理接口，整合分析、匿名化和分割功能。
"""

import logging
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass

from presidio_analyzer import RecognizerResult
from presidio_anonymizer import EngineResult

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor import TextProcessorConfig
from .core import (
    PresidioAnalyzer,
    PresidioAnonymizer,
    TextSegmenter,
    TextSegment
)


@dataclass
class ProcessingResult:
    """文本处理结果"""
    original_text: str
    segments: List[TextSegment]
    analysis_results: List[List[RecognizerResult]]  # 每个段落的分析结果
    anonymized_segments: List[EngineResult]  # 每个段落的匿名化结果
    anonymized_text: str  # 最终的匿名化文本
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'original_text': self.original_text,
            'segments': [seg.to_dict() for seg in self.segments],
            'analysis_results': [
                [result.to_dict() for result in results] 
                for results in self.analysis_results
            ],
            'anonymized_segments': [
                {
                    'text': result.text,
                    'items': [
                        {
                            'start': item.start,
                            'end': item.end,
                            'entity_type': item.entity_type,
                            'text': item.text,
                            'operator': item.operator
                        } for item in result.items
                    ]
                } for result in self.anonymized_segments
            ],
            'anonymized_text': self.anonymized_text,
            'metadata': self.metadata or {}
        }


class TextProcessor:
    """文本处理器主类
    
    整合PII分析、匿名化和文本分割功能，提供统一的处理接口。
    """
    
    def __init__(self, config: Optional[TextProcessorConfig] = None):
        """
        初始化文本处理器
        
        Args:
            config: 文本处理器配置，如果为None则使用默认配置
        """
        self.config = config or TextProcessorConfig()
        self.logger = get_module_logger(__name__)
        # 初始化核心组件
        self._analyzer: Optional[PresidioAnalyzer] = None
        self._anonymizer: Optional[PresidioAnonymizer] = None
        self._segmenter: Optional[TextSegmenter] = None

        self._setup_components()

    def _setup_components(self) -> None:
        """设置核心组件"""
        try:
            # 初始化分析器
            if self.config.analyzer.presidio_enabled:
                self._analyzer = PresidioAnalyzer(self.config)
            else:
                self.logger.warning("No analyzer configured")
            
            # 初始化匿名化器
            if self.config.anonymizer.presidio_enabled:
                self._anonymizer = PresidioAnonymizer(self.config.anonymizer)
            else:
                self.logger.warning("No anonymizer configured")
            
            # 初始化分割器
            self._segmenter = TextSegmenter(self.config.segmentation)

            self.logger.info("Text processor components initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text processor components: {e}")
            raise
    
    @property
    def analyzer(self) -> Optional[PresidioAnalyzer]:
        """获取分析器实例"""
        return self._analyzer

    @property
    def anonymizer(self) -> Optional[PresidioAnonymizer]:
        """获取匿名化器实例"""
        return self._anonymizer

    @property
    def segmenter(self) -> Optional[TextSegmenter]:
        """获取分割器实例"""
        return self._segmenter
    
    def process(
        self,
        text: str,
        enable_segmentation: bool = False,
        enable_analysis: bool = True,
        enable_anonymization: bool = True,
        language: Optional[str] = None,
        entities: Optional[List[str]] = None,
        score_threshold: Optional[float] = None,
        operators: Optional[Dict[str, str]] = None
    ) -> ProcessingResult:
        """
        处理文本，执行完整的PII检测和匿名化流程
        
        Args:
            text: 待处理的文本
            enable_segmentation: 是否启用文本分割
            enable_analysis: 是否启用PII分析
            enable_anonymization: 是否启用匿名化
            language: 语言代码
            entities: 要检测的实体类型列表
            score_threshold: 置信度阈值
            operators: 实体类型到操作符的映射
            
        Returns:
            处理结果
        """
        if not text or not text.strip():
            return ProcessingResult(
                original_text=text,
                segments=[],
                analysis_results=[],
                anonymized_segments=[],
                anonymized_text=text
            )
        
        try:
            # 1. 文本分割
            segments = []
            if enable_segmentation and self._segmenter:
                segments = self._segmenter.segment(text)
                self.logger.debug(f"Text segmented into {len(segments)} segments")
            else:
                # 如果不启用分割或分割器不可用，将整个文本作为一个段落
                segments = [TextSegment(text=text, start=0, end=len(text))]
            
            # 2. PII分析
            analysis_results = []
            if enable_analysis and self._analyzer:
                for segment in segments:
                    segment_results = self._analyzer.analyze(
                        text=segment.text,
                        language=language,
                        entities=entities,
                        score_threshold=score_threshold
                    )
                    analysis_results.append(segment_results)
                    
                total_entities = sum(len(results) for results in analysis_results)
                self.logger.debug(f"Found {total_entities} PII entities across all segments")
            else:
                analysis_results = [[] for _ in segments]
            
            # 3. 匿名化
            anonymized_segments = []
            if enable_anonymization and self._anonymizer:
                for segment, segment_analysis in zip(segments, analysis_results):
                    anonymized_result = self._anonymizer.anonymize(
                        text=segment.text,
                        analyzer_results=segment_analysis,
                        operators=operators
                    )
                    anonymized_segments.append(anonymized_result)
            else:
                # 如果不启用匿名化，返回原始文本
                for segment in segments:
                    anonymized_segments.append(EngineResult(
                        text=segment.text,
                        items=[]
                    ))
            
            # 4. 重建完整的匿名化文本
            anonymized_text = self._reconstruct_text(segments, anonymized_segments)
            
            # 5. 构建结果
            result = ProcessingResult(
                original_text=text,
                segments=segments,
                analysis_results=analysis_results,
                anonymized_segments=anonymized_segments,
                anonymized_text=anonymized_text,
                metadata={
                    'total_segments': len(segments),
                    'total_entities': sum(len(results) for results in analysis_results),
                    'processing_config': {
                        'segmentation_enabled': enable_segmentation,
                        'analysis_enabled': enable_analysis,
                        'anonymization_enabled': enable_anonymization,
                        'language': language,
                        'entities': entities,
                        'score_threshold': score_threshold
                    }
                }
            )
            
            self.logger.info("Text processing completed successfully")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during text processing: {e}")
            # 返回原始文本作为fallback
            return ProcessingResult(
                original_text=text,
                segments=[],
                analysis_results=[],
                anonymized_segments=[],
                anonymized_text=text,
                metadata={'error': str(e)}
            )
    
    def _reconstruct_text(
        self, 
        segments: List[TextSegment], 
        anonymized_segments: List[EngineResult]
    ) -> str:
        """
        重建匿名化后的完整文本
        
        Args:
            segments: 原始文本段落
            anonymized_segments: 匿名化后的段落
            
        Returns:
            重建的完整文本
        """
        if not segments or not anonymized_segments:
            return ""
        
        if len(segments) != len(anonymized_segments):
            self.logger.warning("Segments and anonymized segments count mismatch")
            return "".join(result.text for result in anonymized_segments)
        
        # 如果只有一个段落，直接返回
        if len(segments) == 1:
            return anonymized_segments[0].text
        
        # 多个段落的情况，需要考虑原始文本中的间隔
        result_text = ""
        last_end = 0
        
        for i, (segment, anonymized) in enumerate(zip(segments, anonymized_segments)):
            # 添加段落之间的原始文本（如空白字符等）
            if segment.start > last_end:
                gap_text = segments[0].text[last_end:segment.start] if i == 0 else ""
                result_text += gap_text
            
            # 添加匿名化后的段落文本
            result_text += anonymized.text
            last_end = segment.end
        
        return result_text
    
    def batch_process(
        self,
        texts: List[str],
        **kwargs
    ) -> List[ProcessingResult]:
        """
        批量处理文本
        
        Args:
            texts: 待处理的文本列表
            **kwargs: 其他参数传递给process方法
            
        Returns:
            处理结果列表
        """
        results = []
        for i, text in enumerate(texts):
            try:
                result = self.process(text, **kwargs)
                results.append(result)
                self.logger.debug(f"Processed text {i+1}/{len(texts)}")
            except Exception as e:
                self.logger.error(f"Error processing text {i+1}: {e}")
                # 添加错误结果
                results.append(ProcessingResult(
                    original_text=text,
                    segments=[],
                    analysis_results=[],
                    anonymized_segments=[],
                    anonymized_text=text,
                    metadata={'error': str(e)}
                ))
        
        self.logger.info(f"Batch processing completed: {len(results)} texts processed")
        return results
    
    def analyze_only(
        self,
        text: str,
        language: Optional[str] = None,
        entities: Optional[List[str]] = None,
        score_threshold: Optional[float] = None
    ) -> List[RecognizerResult]:
        """
        仅执行PII分析，不进行匿名化
        
        Args:
            text: 待分析的文本
            language: 语言代码
            entities: 要检测的实体类型列表
            score_threshold: 置信度阈值
            
        Returns:
            分析结果列表
        """
        if not self._analyzer:
            self.logger.warning("Analyzer not available")
            return []
        
        return self._analyzer.analyze(text, language, entities, score_threshold)
    
    def anonymize_only(
        self,
        text: str,
        analyzer_results: List[RecognizerResult],
        operators: Optional[Dict[str, str]] = None
    ) -> EngineResult:
        """
        仅执行匿名化，使用预先分析的结果
        
        Args:
            text: 原始文本
            analyzer_results: 预先分析的结果
            operators: 实体类型到操作符的映射
            
        Returns:
            匿名化结果
        """
        if not self._anonymizer:
            self.logger.warning("Anonymizer not available")
            return EngineResult(text=text, items=[])
        
        return self._anonymizer.anonymize(text, analyzer_results, operators)
    
    def segment_only(self, text: str) -> List[TextSegment]:
        """
        仅执行文本分割
        
        Args:
            text: 待分割的文本
            
        Returns:
            文本段落列表
        """
        if not self._segmenter:
            self.logger.warning("Segmenter not available")
            return [TextSegment(text=text, start=0, end=len(text))]
        
        return self._segmenter.segment(text)
    
    def get_supported_entities(self) -> List[str]:
        """
        获取支持的实体类型列表
        
        Returns:
            支持的实体类型列表
        """
        if not self._analyzer:
            return []
        
        return self._analyzer.get_supported_entities()
    
    def get_supported_operators(self) -> List[str]:
        """
        获取支持的匿名化操作符列表
        
        Returns:
            支持的操作符列表
        """
        # 返回Presidio支持的标准操作符
        return ["replace", "mask", "hash", "redact", "encrypt", "custom"]
