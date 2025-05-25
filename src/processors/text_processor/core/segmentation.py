#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本分割器模块

提供文本分割功能，将长文本分割为适合处理的段落或句子。
"""

import re
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from functools import lru_cache

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor import SegmentationConfig


@dataclass
class TextSegment:
    """文本段落"""
    text: str
    start: int
    end: int
    segment_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'text': self.text,
            'start': self.start,
            'end': self.end,
            'segment_id': self.segment_id,
            'metadata': self.metadata or {}
        }


class BaseSegmenter(ABC):
    """文本分割器基类
    
    定义了文本分割器的标准接口，所有具体的分割器实现都应该继承此类。
    """
    
    def __init__(self, config: SegmentationConfig):
        """
        初始化分割器
        
        Args:
            config: 分割器配置
        """
        self.config = config
        self.logger = get_module_logger(__name__)
        self._setup_segmenter()
    
    @abstractmethod
    def _setup_segmenter(self) -> None:
        """
        设置分割器
        子类需要实现此方法来初始化具体的分割器
        """
        pass
    
    @abstractmethod
    def segment(self, text: str) -> List[TextSegment]:
        """
        分割文本
        
        Args:
            text: 待分割的文本
            
        Returns:
            文本段落列表
        """
        pass
    
    def batch_segment(self, texts: List[str]) -> List[List[TextSegment]]:
        """
        批量分割文本
        
        Args:
            texts: 待分割的文本列表
            
        Returns:
            每个文本的段落列表
        """
        results = []
        for text in texts:
            try:
                result = self.segment(text)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Error segmenting text: {e}")
                # 返回整个文本作为一个段落
                results.append([TextSegment(text=text, start=0, end=len(text))])
        return results
    
    def _post_process_segments(self, segments: List[TextSegment]) -> List[TextSegment]:
        """
        后处理分割结果
        
        Args:
            segments: 原始分割结果
            
        Returns:
            后处理后的分割结果
        """
        processed_segments = []
        
        for segment in segments:
            # 移除空段落
            if self.config.post_processing["remove_empty_segments"] and not segment.text.strip():
                continue
            
            # 去除空白字符
            text = segment.text
            if self.config.post_processing["strip_whitespace"]:
                text = text.strip()
            
            # 长度过滤
            min_length = self.config.post_processing["min_segment_length"]
            max_length = self.config.post_processing["max_segment_length"]
            
            if len(text) < min_length or len(text) > max_length:
                continue
            
            # 更新段落文本
            processed_segment = TextSegment(
                text=text,
                start=segment.start,
                end=segment.end,
                segment_id=segment.segment_id,
                metadata=segment.metadata
            )
            processed_segments.append(processed_segment)
        
        return processed_segments


class TextSegmenter(BaseSegmenter):
    """文本分割器实现类
    
    支持多种分割策略：句子分割、段落分割、固定长度分割和自定义分割。
    """
    
    def __init__(self, config: SegmentationConfig):
        """
        初始化文本分割器
        
        Args:
            config: 分割器配置
        """
        self._nlp = None
        super().__init__(config)
    
    def _setup_segmenter(self) -> None:
        """设置文本分割器"""
        try:
            # 根据配置初始化相应的分割器
            if self.config.segmentation_strategy == "sentence":
                self._setup_sentence_segmenter()
            
            self.logger.info(f"Text segmenter initialized with strategy: {self.config.segmentation_strategy}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize text segmenter: {e}")
            raise
    
    def _setup_sentence_segmenter(self) -> None:
        """设置句子分割器"""
        try:
            sentence_config = self.config.sentence_segmentation
            if sentence_config.get("use_spacy", False):
                import spacy
                model_name = sentence_config.get("spacy_model", "zh_core_web_sm")
                try:
                    self._nlp = spacy.load(model_name)
                    self.logger.info(f"Loaded spaCy model: {model_name}")
                except OSError:
                    self.logger.warning(f"spaCy model {model_name} not found, using pattern-based segmentation")
                    self._nlp = None
        except Exception as e:
            self.logger.warning(f"Failed to setup spaCy: {e}, using pattern-based segmentation")
            self._nlp = None
    
    def segment(self, text: str) -> List[TextSegment]:
        """
        分割文本
        
        Args:
            text: 待分割的文本
            
        Returns:
            文本段落列表
        """
        if not text or not text.strip():
            return []
        
        strategy = self.config.segmentation_strategy
        
        if strategy == "sentence":
            segments = self._segment_by_sentence(text)
        elif strategy == "paragraph":
            segments = self._segment_by_paragraph(text)
        elif strategy == "fixed_length":
            segments = self._segment_by_fixed_length(text)
        elif strategy == "custom":
            segments = self._segment_by_custom(text)
        else:
            self.logger.warning(f"Unknown segmentation strategy: {strategy}, using sentence")
            segments = self._segment_by_sentence(text)
        
        # 后处理
        segments = self._post_process_segments(segments)
        
        # 添加段落ID
        for i, segment in enumerate(segments):
            segment.segment_id = f"seg_{i:04d}"
        
        self.logger.debug(f"Segmented text into {len(segments)} segments")
        return segments
    
    def _segment_by_sentence(self, text: str) -> List[TextSegment]:
        """按句子分割文本"""
        segments = []
        
        if self._nlp:
            # 使用spaCy分割
            doc = self._nlp(text)
            for sent in doc.sents:
                start = sent.start_char
                end = sent.end_char
                sentence_text = text[start:end]
                
                # 长度过滤
                config = self.config.sentence_segmentation
                min_length = config.get("min_sentence_length", 0)
                max_length = config.get("max_sentence_length", float('inf'))
                
                if min_length <= len(sentence_text) <= max_length:
                    segments.append(TextSegment(
                        text=sentence_text,
                        start=start,
                        end=end
                    ))
        else:
            # 使用正则表达式分割
            config = self.config.sentence_segmentation
            patterns = config.get("custom_patterns", [r"[。！？]", r"[.!?]"])
            
            # 合并所有模式
            pattern = "|".join(f"({p})" for p in patterns)
            
            # 分割文本
            parts = re.split(pattern, text)
            current_pos = 0
            current_sentence = ""
            
            for part in parts:
                if part is None:
                    continue
                
                current_sentence += part
                current_pos += len(part)
                
                # 检查是否是句子结束符
                if any(re.match(p, part) for p in patterns):
                    # 创建句子段落
                    start = current_pos - len(current_sentence)
                    end = current_pos
                    
                    if current_sentence.strip():
                        segments.append(TextSegment(
                            text=current_sentence,
                            start=start,
                            end=end
                        ))
                    
                    current_sentence = ""
        
        return segments
    
    def _segment_by_paragraph(self, text: str) -> List[TextSegment]:
        """按段落分割文本"""
        segments = []
        config = self.config.paragraph_segmentation
        separators = config.get("paragraph_separators", ["\n\n", "\r\n\r\n"])
        
        # 使用第一个分隔符进行分割
        separator = separators[0] if separators else "\n\n"
        paragraphs = text.split(separator)
        
        current_pos = 0
        for paragraph in paragraphs:
            if paragraph.strip():
                start = current_pos
                end = current_pos + len(paragraph)
                
                # 长度过滤
                min_length = config.get("min_paragraph_length", 0)
                max_length = config.get("max_paragraph_length", float('inf'))
                
                if min_length <= len(paragraph) <= max_length:
                    segments.append(TextSegment(
                        text=paragraph,
                        start=start,
                        end=end
                    ))
            
            current_pos += len(paragraph) + len(separator)
        
        return segments
    
    def _segment_by_fixed_length(self, text: str) -> List[TextSegment]:
        """按固定长度分割文本"""
        segments = []
        config = self.config.fixed_length_segmentation
        chunk_size = config.get("chunk_size", 500)
        overlap_size = config.get("overlap_size", 50)
        respect_boundaries = config.get("respect_word_boundaries", True)
        
        current_pos = 0
        while current_pos < len(text):
            end_pos = min(current_pos + chunk_size, len(text))
            chunk_text = text[current_pos:end_pos]
            
            # 尊重词边界
            if respect_boundaries and end_pos < len(text):
                # 向后查找最近的空白字符
                for i in range(len(chunk_text) - 1, -1, -1):
                    if chunk_text[i].isspace():
                        chunk_text = chunk_text[:i]
                        end_pos = current_pos + i
                        break
            
            if chunk_text.strip():
                segments.append(TextSegment(
                    text=chunk_text,
                    start=current_pos,
                    end=end_pos
                ))
            
            # 计算下一个位置（考虑重叠）
            current_pos = max(current_pos + 1, end_pos - overlap_size)
        
        return segments
    
    def _segment_by_custom(self, text: str) -> List[TextSegment]:
        """使用自定义方法分割文本"""
        config = self.config.custom_segmentation
        custom_function = config.get("custom_function")
        
        if custom_function and callable(custom_function):
            try:
                return custom_function(text)
            except Exception as e:
                self.logger.error(f"Custom segmentation function failed: {e}")
        
        # 使用自定义模式
        patterns = config.get("custom_patterns", [])
        if patterns:
            segments = []
            # 简单的模式分割实现
            for pattern in patterns:
                parts = re.split(pattern, text)
                current_pos = 0
                for part in parts:
                    if part.strip():
                        start = current_pos
                        end = current_pos + len(part)
                        segments.append(TextSegment(
                            text=part,
                            start=start,
                            end=end
                        ))
                    current_pos += len(part)
            return segments
        
        # 默认返回整个文本作为一个段落
        return [TextSegment(text=text, start=0, end=len(text))]