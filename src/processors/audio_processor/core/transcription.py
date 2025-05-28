#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音转录模块 - 基于 WhisperX

此模块提供高精度的语音转录功能，使用 WhisperX 库进行语音识别和时间戳对齐。
WhisperX 相比原始 Whisper 提供了更准确的词级时间戳和更好的说话人对齐能力。
"""

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch
import whisperx
from whisperx.asr import FasterWhisperPipeline
from whisperx.types import (
    SingleWordSegment,
    TranscriptionResult,
    AlignedTranscriptionResult,
)

from src.commons.loggers import get_module_logger
from src.configs.processors.audio_processor.audio_configs import AudioTranscriptionConfig

# 忽略警告信息
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# 获取模块日志记录器
logger = get_module_logger(__name__)


class WhisperXTranscriber:
    """基于 WhisperX 的语音转录器
    
    使用 WhisperX 库提供高精度的语音转录和词级时间戳对齐功能。
    相比原始 Whisper，提供更准确的时间戳和更好的批处理支持。
    """
    
    def __init__(
            self,
            config: AudioTranscriptionConfig,
            device: str = "cpu"
        ):
        """初始化 WhisperX 转录器
        
        Args:
            config: 转录配置，如果为None则使用默认配置
        """
  
        self.config = config or AudioTranscriptionConfig()
        self.device = torch.device(device)

        # 模型组件
        self._whisper_model: FasterWhisperPipeline | None = None
        self._align_model: Optional[Any] = None
        self._align_metadata: Optional[Dict] = None
        self._is_initialized = False
        
        # 支持的音频格式
        self._supported_formats = ['.wav', '.mp3', '.flac', '.ogg', '.m4a']
    
    def _initialize_models(self) -> None:
        """初始化 WhisperX 模型"""
        if self._is_initialized:
            return
        
        try:
            logger.info(f"正在加载 WhisperX 模型: {self.config.model_size}")
            
            # 加载 Whisper 模型
            self._whisper_model = whisperx.load_model(
                self.config.model_size,
                device=self.device.type,
                compute_type=self.config.compute_type,
                language=self.config.language
            )
            
            logger.info("WhisperX 模型加载成功")
            self._is_initialized = True
            
        except Exception as e:
            error_msg = f"WhisperX 模型初始化失败: {str(e)}"
            logger.error(error_msg)
            raise
    
    def _load_align_model(self, language_code: str) -> None:
        """加载对齐模型
        
        Args:
            language_code: 语言代码
            
        Raises:
            RuntimeError: 对齐模型加载失败时抛出
        """
        try:
            self._align_model, self._align_metadata = whisperx.load_align_model(
                language_code=language_code,
                device=self.device,
                model_name=self.config.align_model
            )
            logger.info(f"对齐模型加载成功: {language_code}")

        except Exception as e:
            error_msg = f"对齐模型加载失败 (语言: {language_code}): {str(e)}"
            logger.error(error_msg)
            # 不再忽略错误，而是抛出异常
            raise RuntimeError(error_msg) from e

    def transcribe(self, audio_path: Union[str, Path]) -> AlignedTranscriptionResult:
        """执行音频转录
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            转录结果
            
        Raises:
            TranscriptionError: 转录失败时抛出
        """
        audio_file = Path(audio_path)
        
        # 检查文件是否存在
        if not audio_file.exists():
            raise FileNotFoundError(f"音频文件不存在: {audio_file}")
        
        logger.info(f"开始转录音频: {audio_file}")
        
        # 确保模型已初始化
        self._initialize_models()
        assert self._whisper_model is not None, "WhisperX 模型未初始化"
        
        try:
            # 加载音频
            audio = whisperx.load_audio(str(audio_file))
            
            # 执行转录
            result = self._whisper_model.transcribe(
                audio,
                batch_size=self.config.batch_size,
                chunk_size=self.config.chunk_size,
                print_progress=False
            )
            logger.info("音频转录完成")
            
            self._load_align_model(result["language"])
            
            aligned_result = whisperx.align(
                result["segments"],
                self._align_model,
                self._align_metadata,
                audio,
                self.device,
                interpolate_method=self.config.interpolate_method,
                return_char_alignments=self.config.return_char_alignments
            )
            logger.info("时间戳对齐完成")
            

            logger.info(f"转录完成，检测语言: {self.config.language}")
            return aligned_result
            
        except Exception as e:
            error_msg = f"音频转录失败: {str(e)}"
            logger.error(error_msg)
            raise
    