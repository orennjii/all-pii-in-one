#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频PII检测模块 - 个人身份信息检测

此模块提供音频中个人身份信息(PII)的检测功能，结合转录和说话人分割进行文本分析。
支持多种PII实体类型检测、说话人关联分析、置信度评估等功能。
"""

import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass

from presidio_analyzer import RecognizerResult
from whisperx.types import AlignedTranscriptionResult, SingleAlignedSegment

from src.commons.loggers import get_module_logger
from src.processors.text_processor.processor import TextProcessor, ProcessingResult
from src.configs.processors.text_processor import TextProcessorConfig
from .diarization import DiarizationResult, SpeakerSegment

# 获取模块日志记录器
logger = get_module_logger(__name__)


@dataclass
class PIIEntity:
    """PII实体信息 - 简化版"""
    entity_type: str  # 实体类型
    text       : str  # 实体文本
    confidence : float  # 置信度分数
    start      : Optional[int]   = None  # 实体在文本中的起始字符位置
    end        : Optional[int]   = None  # 实体在文本中的结束字符位置
    start_time : Optional[float] = None  # 音频开始时间
    end_time   : Optional[float] = None  # 音频结束时间
    speaker_id : Optional[str]   = None  # 说话人标识符


@dataclass
class AudioPIIResult:
    """音频PII检测结果 - 统一简化版"""
    text           : str  # 原始转录文本
    anonymized_text: str  # 匿名化后的文本
    pii_entities   : List[PIIEntity]  # 检测到的PII实体列表
    speakers       : Optional[Dict[str, List[PIIEntity]]] = None  # 按说话人分组的PII实体
    
    @property
    def pii_count(self) -> int:
        """PII实体总数"""
        return len(self.pii_entities)
    
    @property
    def avg_confidence(self) -> float:
        """平均置信度"""
        if not self.pii_entities:
            return 0.0
        return sum(e.confidence for e in self.pii_entities) / len(self.pii_entities)
    
    def get_entities_by_type(self, entity_type: str) -> List[PIIEntity]:
        """获取指定类型的PII实体"""
        return [e for e in self.pii_entities if e.entity_type == entity_type]


class AudioPIIDetector:
    """音频PII检测器
    
    结合转录、说话人分割和文本分析，检测音频中的个人身份信息。
    支持多种PII实体类型、实时检测和说话人关联分析。
    """
    
    def __init__(
        self,
        config: TextProcessorConfig,
        confidence_threshold: float = 0.5,
        language: str = 'zh'
    ):
        """初始化PII检测器
        
        Args:
            config: 文本处理器配置
            confidence_threshold: 置信度阈值
            language: 语言代码
        """
        self.config = config
        self.entity_types = self.config.supported_entities
        self.confidence_threshold = confidence_threshold
        self.language = language

        self.text_processor = TextProcessor(self.config)
        
        logger.info(f"音频PII检测器初始化完成，检测实体类型: {self.entity_types}")
    
    def detect(
        self,
        transcription_result: AlignedTranscriptionResult,
        diarization_result: Optional[DiarizationResult] = None,
        **kwargs
    ) -> AudioPIIResult:
        """检测音频中的PII
        
        Args:
            transcription_result: WhisperX 对齐转录结果
            diarization_result: 说话人分割结果
            **kwargs: 其他参数
            
        Returns:
            PII检测结果
        """
        try:
            logger.info("开始音频PII检测")
            
            # 合并所有片段文本
            full_text = " ".join([segment["text"] for segment in transcription_result["segments"]])
            # 使用完整文本进行PII检测，然后映射回音频时间戳
            pii_entities, analysis_results = self._detect_pii_with_audio_timestamps(
                transcription_result,
                kwargs.get('language', self.language),
                kwargs.get('entities', self.entity_types),
                kwargs.get('score_threshold', self.confidence_threshold)
            )
            
            # 生成匿名化文本
            assert self.text_processor.anonymizer, "文本匿名化器未初始化"
            anonymized_text = self.text_processor.anonymizer.anonymize(
                text=full_text,
                analyzer_results=analysis_results
            )
            
            # 关联说话人信息
            if diarization_result:
                self._associate_speakers(pii_entities, diarization_result)
            
            # 按说话人分组PII实体
            speakers = self._group_entities_by_speaker(pii_entities) if diarization_result else None
            
            # 构建结果
            result = AudioPIIResult(
                text=full_text,
                anonymized_text=anonymized_text.text,
                pii_entities=pii_entities,
                speakers=speakers
            )
            
            logger.info(f"PII检测完成，找到 {len(pii_entities)} 个PII实体")
            return result
            
        except Exception as e:
            logger.error(f"音频PII检测失败: {e}")
            raise

    def _associate_speakers(
        self,
        pii_entities: List[PIIEntity],
        diarization_result: DiarizationResult
    ) -> None:
        """为PII实体关联说话人信息（就地修改）"""
        for entity in pii_entities:
            if entity.start_time is not None and entity.end_time is not None:
                center_time = (entity.start_time + entity.end_time) / 2
                
                # 找到包含此时间点的说话人片段
                for speaker_segment in diarization_result.segments:
                    if (speaker_segment.start_time <= center_time <= speaker_segment.end_time):
                        entity.speaker_id = speaker_segment.speaker_id
                        break
    
    def _group_entities_by_speaker(self, pii_entities: List[PIIEntity]) -> Dict[str, List[PIIEntity]]:
        """按说话人分组PII实体"""
        speaker_groups = {}
        
        for entity in pii_entities:
            speaker_id = entity.speaker_id or 'unknown'
            
            if speaker_id not in speaker_groups:
                speaker_groups[speaker_id] = []
            
            speaker_groups[speaker_id].append(entity)
        
        return speaker_groups
    
    def _detect_pii_with_audio_timestamps(
        self,
        transcription_result: AlignedTranscriptionResult,
        language: str,
        entities: List[str],
        score_threshold: float
    ) -> Tuple[List[PIIEntity], List[RecognizerResult]]:
        """使用WhisperX的word-level时间戳进行精确PII检测和时间映射
        
        Args:
            transcription_result: WhisperX对齐转录结果
            language: 语言代码
            entities: 要检测的实体类型
            score_threshold: 置信度阈值
            
        Returns:
            包含精确音频时间戳的PII实体列表
        """
        logger.info("开始基于word-level时间戳的PII检测")
        
        # 1. 构建完整文本和字符位置映射
        full_text = ""
        char_to_word_map = []  # 每个字符对应的word信息
        
        for segment in transcription_result["segments"]:
            segment_start_char = len(full_text)
            segment_text = segment["text"]
            
            # 检查segment是否有words字段
            if "words" in segment and segment["words"]:
                words = segment["words"]
                
                # 当前在segment文本中的位置
                current_pos = 0
                
                # 为segment文本中的每个字符建立到word的映射
                for word_info in words:
                    word_text = word_info["word"]
                    word_start = word_info["start"]
                    word_end = word_info["end"]
                    
                    # 在segment文本中找到这个word的位置
                    word_start_in_segment = segment_text.find(word_text, current_pos)
                    if word_start_in_segment != -1:
                        # 为之前没有映射的字符填充（通常是空格或标点）
                        while len(char_to_word_map) < len(full_text) + word_start_in_segment:
                            char_to_word_map.append({
                                "word_text": "",
                                "word_start_time": word_start,
                                "word_end_time": word_start,
                                "char_index_in_word": 0
                            })
                        
                        # 为这个word的每个字符创建映射
                        for i in range(len(word_text)):
                            char_to_word_map.append({
                                "word_text": word_text,
                                "word_start_time": word_start,
                                "word_end_time": word_end,
                                "char_index_in_word": i
                            })
                        
                        current_pos = word_start_in_segment + len(word_text)
                    else:
                        # 如果找不到word，为word的每个字符创建默认映射
                        for i in range(len(word_text)):
                            char_to_word_map.append({
                                "word_text": word_text,
                                "word_start_time": word_start,
                                "word_end_time": word_end,
                                "char_index_in_word": i
                            })
            else:
                # 如果没有words字段，使用segment级别的时间戳
                for char in segment_text:
                    char_to_word_map.append({
                        "word_text": segment_text,
                        "word_start_time": segment.get("start", 0.0),
                        "word_end_time": segment.get("end", 0.0),
                        "char_index_in_word": 0
                    })
            
            full_text += segment_text
            
            # 添加分段间的空格
            if len(transcription_result["segments"]) > 1:
                full_text += " "
                char_to_word_map.append({
                    "word_text": " ",
                    "word_start_time": segment.get("end", 0.0),
                    "word_end_time": segment.get("end", 0.0),
                    "char_index_in_word": 0
                })
        
        full_text = full_text.strip()
        
        # 确保char_to_word_map的长度与full_text匹配
        while len(char_to_word_map) < len(full_text):
            char_to_word_map.append({
                "word_text": "",
                "word_start_time": 0.0,
                "word_end_time": 0.0,
                "char_index_in_word": 0
            })
        
        # 2. 使用text_processor检测PII
        assert self.text_processor.analyzer, "文本分析器未初始化"        
        analysis_results = self.text_processor.analyzer.analyze(
            text=full_text,
            language=language,
            entities=entities,
            score_threshold=score_threshold
        )
        
        # 3. 为每个检测到的PII实体映射精确的音频时间戳
        pii_entities = []
        
        for result in analysis_results:
            entity_text = full_text[result.start:result.end]
            
            # 获取实体开始和结束字符位置的时间戳
            start_char_idx = result.start
            end_char_idx = min(result.end - 1, len(char_to_word_map) - 1)  # 确保不超出范围
            
            # 确保索引在有效范围内
            if start_char_idx < len(char_to_word_map) and end_char_idx >= 0:
                start_word_info = char_to_word_map[start_char_idx]
                end_word_info = char_to_word_map[end_char_idx]
                
                # 计算精确的开始和结束时间
                start_time = start_word_info["word_start_time"]
                end_time = end_word_info["word_end_time"]
                
                # 如果开始和结束在同一个word内，可以进行更精确的时间计算
                if (start_word_info["word_text"] == end_word_info["word_text"] and 
                    start_word_info["word_start_time"] == end_word_info["word_start_time"] and
                    start_word_info["word_text"]):  # 确保不是空字符串
                    
                    word_duration = end_word_info["word_end_time"] - start_word_info["word_start_time"]
                    word_length = len(start_word_info["word_text"])
                    
                    if word_length > 0:
                        char_duration = word_duration / word_length
                        start_time = start_word_info["word_start_time"] + (start_word_info["char_index_in_word"] * char_duration)
                        end_time = start_word_info["word_start_time"] + ((end_word_info["char_index_in_word"] + 1) * char_duration)
                
                entity = PIIEntity(
                    entity_type=result.entity_type,
                    text=entity_text,
                    confidence=result.score,
                    start=start_char_idx,
                    end=end_char_idx,
                    start_time=start_time,
                    end_time=end_time
                )
                
                pii_entities.append(entity)
                
                logger.debug(f"检测到PII实体: {entity.entity_type} = '{entity.text}' "
                           f"at {entity.start_time:.2f}s - {entity.end_time:.2f}s (置信度: {entity.confidence:.2f})")
            else:
                logger.warning(f"字符索引超出范围: start={start_char_idx}, end={end_char_idx}, map_length={len(char_to_word_map)}")
        
        logger.info(f"完成PII检测，共找到 {len(pii_entities)} 个实体")
        return pii_entities, analysis_results
    
    def _generate_anonymized_text(self, original_text: str, pii_entities: List[PIIEntity]) -> str:
        """生成匿名化文本
        
        Args:
            original_text: 原始文本
            pii_entities: 检测到的PII实体列表
            
        Returns:
            匿名化后的文本
        """
        if not pii_entities or not self.text_processor.anonymizer:
            return original_text
        
        try:
            # 重新分析文本以获取RecognizerResult对象，这是anonymizer需要的
            if self.text_processor.analyzer:
                analysis_results = self.text_processor.analyzer.analyze(
                    text=original_text,
                    language=self.language,
                    entities=self.entity_types,
                    score_threshold=self.confidence_threshold
                )
                
                if analysis_results:
                    anonymized_result = self.text_processor.anonymizer.anonymize(
                        text=original_text,
                        analyzer_results=analysis_results
                    )
                    return anonymized_result.text
            
            return original_text
            
        except Exception as e:
            logger.warning(f"文本匿名化失败: {e}")
            return original_text