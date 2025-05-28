#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
说话人分割模块 - 基于 Pyannote Audio

此模块提供音频说话人分割功能，使用最新的 Pyannote Audio 模型进行说话人识别和时间分割。
支持自动检测说话人数量、重叠语音处理、以及说话人时间线生成。
"""

import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torchaudio
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment, Timeline

from src.commons.loggers import get_module_logger
from src.configs.base_config import BaseConfig
from src.configs.processors.audio_processor.audio_configs import AudioDiarizationConfig

# 获取模块日志记录器
logger = get_module_logger(__name__)


@dataclass
class SpeakerSegment:
    """说话人片段数据类 - 简化版"""
    speaker_id: str  # 说话人标识符
    start_time: float  # 开始时间（秒）
    end_time: float  # 结束时间（秒）
    
    @property
    def duration(self) -> float:
        """片段时长（秒）"""
        return self.end_time - self.start_time


@dataclass  
class DiarizationResult:
    """说话人分割结果 - 简化版"""
    segments: List[SpeakerSegment]  # 说话人片段列表
    speaker_count: int  # 说话人总数
    
    @property
    def total_duration(self) -> float:
        """音频总时长"""
        if not self.segments:
            return 0.0
        return max(segment.end_time for segment in self.segments)
    
    def get_speaker_timeline(self, speaker_id: str) -> List[Tuple[float, float]]:
        """获取指定说话人的时间线"""
        timeline = []
        for segment in self.segments:
            if segment.speaker_id == speaker_id:
                timeline.append((segment.start_time, segment.end_time))
        return sorted(timeline)
    
    def get_speaking_time(self, speaker_id: str) -> float:
        """获取指定说话人的总说话时长"""
        total_time = 0.0
        for segment in self.segments:
            if segment.speaker_id == speaker_id:
                total_time += segment.duration
        return total_time

class PyannoteAudioDiarizer():
    """基于 Pyannote Audio 的说话人分割器
    
    使用最新的 Pyannote Audio 模型进行高精度说话人分割，
    支持自动说话人数量检测和重叠语音处理。
    """
    
    def __init__(
        self, 
        config: Optional[AudioDiarizationConfig] = None,
        device: str = 'cpu',
    ):
        """初始化 Pyannote 说话人分割器
        
        Args:
            config: 分割配置，如果为None则使用默认配置
            auth_token: Hugging Face 访问令牌（某些模型需要）
        """
        self.config = config or AudioDiarizationConfig()
        self.auth_token = os.getenv('HUGGINGFACE_ACCESS_TOKEN') or self.config.auth_token
        
        # 设置设备
        self.device = torch.device(device)
        logger.info(f"初始化 Pyannote 说话人分割器，使用设备: {self.device.type}")
        
        self._pipeline: Pipeline
        self._is_initialized = False
        
        # 支持的音频格式
        self._supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    def _initialize_pipeline(self) -> None:
        """初始化 Pyannote 流水线"""
        if self._is_initialized:
            return
        
        try:
            logger.info(f"正在加载 Pyannote 模型: {self.config.model}")
            
            # 加载预训练的说话人分割流水线
            self._pipeline = Pipeline.from_pretrained(
                self.config.model,
                use_auth_token=self.auth_token
            )
            
            self._pipeline = self._pipeline.to(self.device)
            
            self._is_initialized = True
            logger.info("Pyannote 流水线初始化成功")
            
        except Exception as e:
            error_msg = f"Pyannote 流水线初始化失败: {str(e)}"
            logger.error(error_msg)
    
    def diarize(self, audio_path: Union[str, Path]) -> DiarizationResult:
        """执行说话人分割
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            说话人分割结果
            
        Raises:
            DiarizationError: 分割失败时抛出
        """
        audio_file = Path(audio_path)
        
        # 检查文件是否存在
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        
        logger.info(f"开始说话人分割: {audio_file}")
        
        # 确保流水线已初始化
        self._initialize_pipeline()
        
        try:
            # 执行说话人分割
            diarization = self._pipeline(
                str(audio_file),
                min_speakers=self.config.min_speakers,
                max_speakers=self.config.max_speakers,
            )

            # 处理分割结果
            result = self._process_diarization_result(diarization, audio_file)

            logger.info(f"说话人分割完成，检测到 {result.speaker_count} 个说话人")
            return result
            
        except Exception as e:
            error_msg = f"说话人分割处理失败: {str(e)}"
            logger.error(error_msg)
            raise

    def _process_diarization_result(
        self, 
        diarization: Annotation, 
        audio_file: Path
    ) -> DiarizationResult:
        """处理说话人分割结果 - 简化版
        
        Args:
            diarization: Pyannote 分割结果
            audio_file: 音频文件路径
            
        Returns:
            处理后的分割结果
        """
        segments = []
        speakers = set()
        
        # 处理每个说话人片段
        for segment, _, speaker in diarization.itertracks(yield_label=True): # type: ignore
            # 过滤过短的片段
            if segment.duration < self.config.min_segment_duration:
                continue

            # 确保说话人ID是字符串
            if not isinstance(speaker, str):
                speaker = str(speaker)
            
            speakers.add(speaker)
            
            # 创建说话人片段 - 简化版
            speaker_segment = SpeakerSegment(
                speaker_id=speaker,
                start_time=segment.start,
                end_time=segment.end
            )
            segments.append(speaker_segment)
        
        # 按时间排序片段
        segments.sort(key=lambda x: x.start_time)
        
        return DiarizationResult(
            segments=segments,
            speaker_count=len(speakers)
        )