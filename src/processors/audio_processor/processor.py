#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理器主模块

提供统一的音频PII处理接口，整合转录、说话人分离、PII检测和匿名化功能。
"""

import logging
import tempfile
import time
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from dataclasses import dataclass

from src.commons.loggers import get_module_logger
from src.configs.processors.audio_processor import AudioProcessorConfig
from src.configs import (
    AppConfig,
    TextProcessorConfig,
)
from whisperx.types import AlignedTranscriptionResult
from .utils.audio_utils import AudioUtils, load_audio_file, validate_audio
from .core.transcription import WhisperXTranscriber, TranscriptionResult
from .core.diarization import PyannoteAudioDiarizer, DiarizationResult
from .core.pii_detection import AudioPIIDetector, AudioPIIResult, PIIEntity
from .core.voice_converter import VoiceConverter  # 保持现有的匿名化实现
from .core.voice_anonymizer import VoiceAnonymizer
from src.configs.processors.audio_processor.voice_anonymizer_config import VoiceAnonymizerConfig


@dataclass
class AudioProcessingResult:
    """音频处理结果"""
    original_audio_path: str
    transcription_result: AlignedTranscriptionResult
    diarization_result: Optional[DiarizationResult]
    pii_detection_result: Optional[AudioPIIResult]
    anonymized_audio_path: Optional[str]


class AudioProcessor:
    """音频处理器
    
    整合转录、说话人分离、PII检测和音频匿名化功能的主要处理器。
    """
    
    def __init__(self, config: AppConfig):
        """初始化音频处理器
        
        Args:
            config: 音频处理器配置
        """
        self.logger = get_module_logger(__name__)
        self.audio_config = config.processor.audio_processor or AudioProcessorConfig()
        self.text_config = config.processor.text_processor or TextProcessorConfig()
        self.audio_utils = AudioUtils()
        
        # 初始化组件
        self._transcriber: Optional[WhisperXTranscriber] = None
        self._diarizer: Optional[PyannoteAudioDiarizer] = None
        self._pii_detector: Optional[AudioPIIDetector] = None
        self._voice_converter: Optional[VoiceConverter] = None
        self._voice_anonymizer: Optional[VoiceAnonymizer] = None
        
        self._setup_components()
        
        self.logger.info("音频处理器初始化完成")
    
    def _setup_components(self) -> None:
        """设置核心组件"""
        try:
            # 初始化转录器
            self._transcriber = WhisperXTranscriber(self.audio_config.transcription)
            
            # 初始化说话人分离器（如果启用）
            if self.audio_config.diarization.enabled:
                self._diarizer = PyannoteAudioDiarizer(self.audio_config.diarization)
            
            # 初始化PII检测器（如果启用）
            if self.text_config:
                self._pii_detector = AudioPIIDetector(
                    config=self.text_config,
                    confidence_threshold=self.text_config.analyzer.default_score_threshold,
                    language=self.text_config.analyzer.default_language
                )
            
            # 初始化语音转换器（如果启用匿名化）
            from src.commons.device_config import get_device
            self._voice_converter = VoiceConverter(device=get_device())
            
            # 初始化语音匿名化器（如果启用）
            if self.audio_config.voice_anonymizer.enabled:
                self._voice_anonymizer = VoiceAnonymizer(self.audio_config.voice_anonymizer)
            
            self.logger.info("音频处理器组件初始化成功")
            
        except Exception as e:
            self.logger.error(f"音频处理器组件初始化失败: {e}")
            raise
    
    @property
    def transcriber(self) -> Optional[WhisperXTranscriber]:
        """获取转录器实例"""
        return self._transcriber
    
    @property
    def diarizer(self) -> Optional[PyannoteAudioDiarizer]:
        """获取说话人分离器实例"""
        return self._diarizer
    
    @property
    def pii_detector(self) -> Optional[AudioPIIDetector]:
        """获取PII检测器实例"""
        return self._pii_detector
    
    @property
    def voice_converter(self) -> Optional[VoiceConverter]:
        """获取语音转换器实例"""
        return self._voice_converter
    
    @property
    def voice_anonymizer(self) -> Optional[VoiceAnonymizer]:
        """获取语音匿名化器实例"""
        return self._voice_anonymizer
    
    def process(
        self,
        audio_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        **kwargs
    ) -> AudioProcessingResult:
        """
        处理音频文件，执行完整的PII检测和匿名化流程
        
        Args:
            audio_path: 音频文件路径
            enable_diarization: 是否启用说话人分离
            enable_pii_detection: 是否启用PII检测
            enable_anonymization: 是否启用音频匿名化
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            音频处理结果
        """
        try:
            start_time = time.time()
            self.logger.info(f"开始处理音频文件: {audio_path}")
            
            # 验证音频文件
            audio_path = Path(audio_path)
            if not validate_audio(audio_path):
                raise ValueError(f"无效的音频文件: {audio_path}")
            
            # 设置处理选项（使用配置默认值）
            
            # 步骤1: 语音转录
            transcription_result = self._transcribe_audio(audio_path, **kwargs)
            
            # 步骤2: 说话人分离（可选）
            diarization_result = self._diarize_audio(audio_path, **kwargs)
            
            # 步骤3: PII检测（可选）
            pii_detection_result = self._detect_pii(
                transcription_result, diarization_result, **kwargs
            )
            
            # 步骤4: 音频匿名化（可选）
            anonymized_audio_path = None
            if pii_detection_result and pii_detection_result.pii_entities:
                anonymized_audio_path = self._anonymize_audio_pii(
                    audio_path, pii_detection_result, output_dir, **kwargs
                )
            
            # 构建处理结果 - 简化版
            processing_time = time.time() - start_time
            
            result = AudioProcessingResult(
                original_audio_path=str(audio_path),
                transcription_result=transcription_result,
                diarization_result=diarization_result,
                pii_detection_result=pii_detection_result,
                anonymized_audio_path=anonymized_audio_path
            )
            
            self.logger.info(f"音频处理完成，用时 {processing_time:.2f} 秒")
            return result
            
        except Exception as e:
            self.logger.error(f"音频处理失败: {e}")
            raise
    
    def _transcribe_audio(
        self,
        audio_path: Path,
        **kwargs
    ) -> AlignedTranscriptionResult:
        """
        转录音频
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 转录参数
            
        Returns:
            转录结果
        """
        try:
            self.logger.info("开始语音转录")
            
            if not self._transcriber:
                raise RuntimeError("转录器未初始化")
                        
            # 执行转录
            result = self._transcriber.transcribe(
                audio_path=audio_path,
                **kwargs
            )
            
            self.logger.info(f"语音转录完成，检测到语言: {result.get('language', 'unknown')}")
            return result
            
        except Exception as e:
            self.logger.error(f"语音转录失败: {e}")
            raise
    
    def _diarize_audio(
        self,
        audio_path: Path,
        **kwargs
    ) -> DiarizationResult:
        """
        说话人分离
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 分离参数
            
        Returns:
            说话人分离结果
        """
        try:
            self.logger.info("开始说话人分离")
            
            if not self._diarizer:
                raise RuntimeError("说话人分离器未初始化")
                        
            # 执行说话人分离
            result = self._diarizer.diarize(
                audio_path=audio_path,
            )
            
            self.logger.info(f"说话人分离完成，检测到 {result.speaker_count} 个说话人")
            return result
            
        except Exception as e:
            self.logger.error(f"说话人分离失败: {e}")
            raise
    
    def _detect_pii(
        self,
        transcription_result: AlignedTranscriptionResult,
        diarization_result: Optional[DiarizationResult],
        **kwargs
    ) -> AudioPIIResult:
        """
        检测PII - 简化版
        
        Args:
            transcription_result: 转录结果
            diarization_result: 说话人分离结果
            **kwargs: 检测参数
            
        Returns:
            PII检测结果
        """
        try:
            self.logger.info("开始PII检测")
            
            if not self._pii_detector:
                raise RuntimeError("PII检测器未初始化")
            
            # 执行PII检测
            result = self._pii_detector.detect(
                transcription_result=transcription_result,
                diarization_result=diarization_result,
            )
            
            self.logger.info(f"PII检测完成，发现 {result.pii_count} 个PII实体")
            return result
            
        except Exception as e:
            self.logger.error(f"PII检测失败: {e}")
            raise
    
    def _anonymize_audio(
        self,
        audio_path: Path,
        diarization_result: Optional[DiarizationResult],
        output_dir: Optional[Path],
        **kwargs
    ) -> str:
        """
        匿名化音频
        
        Args:
            audio_path: 音频文件路径
            diarization_result: 说话人分离结果
            output_dir: 输出目录
            **kwargs: 匿名化参数
            
        Returns:
            匿名化后的音频文件路径
        """
        try:
            self.logger.info("开始音频匿名化")
            
            if not self._voice_converter:
                raise RuntimeError("语音转换器未初始化")
            
            # 设置输出路径
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"anonymized_{audio_path.name}"
            else:
                output_path = audio_path.parent / f"anonymized_{audio_path.name}"
            
            # TODO: 实现基于说话人分离的音频匿名化
            # 这里需要根据说话人分离结果，对每个说话人的音频片段进行声音转换
            # 暂时返回原始音频路径作为占位符
            
            self.logger.warning("音频匿名化功能正在开发中，返回原始音频路径")
            return str(audio_path)
            
        except Exception as e:
            self.logger.error(f"音频匿名化失败: {e}")
            raise
    
    def _anonymize_audio_pii(
        self,
        audio_path: Path,
        pii_detection_result: Optional[AudioPIIResult],
        output_dir: Optional[Union[str, Path]],
        **kwargs
    ) -> Optional[str]:
        """
        基于PII检测结果进行音频匿名化
        
        Args:
            audio_path: 音频文件路径
            pii_detection_result: PII检测结果
            output_dir: 输出目录
            **kwargs: 其他参数
            
        Returns:
            匿名化后的音频文件路径
        """
        try:
            if not pii_detection_result or not pii_detection_result.pii_entities:
                self.logger.info("未检测到PII实体，跳过音频匿名化")
                return None
                
            if not self._voice_anonymizer:
                self.logger.warning("语音匿名化器未初始化，跳过音频匿名化")
                return None
                
            self.logger.info("开始基于PII的音频匿名化")
            
            # 设置输出路径
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                output_path = output_dir / f"anonymized_{audio_path.name}"
            else:
                output_path = audio_path.parent / f"anonymized_{audio_path.name}"
            
            # 执行匿名化
            anonymized_path = self._voice_anonymizer.anonymize_audio(
                audio_path=str(audio_path),
                pii_result=pii_detection_result,
                output_path=str(output_path)
            )
            
            self.logger.info(f"音频匿名化完成: {anonymized_path}")
            self.logger.info(f"处理了 {len(pii_detection_result.pii_entities)} 个PII实体")
            return anonymized_path
                
        except Exception as e:
            self.logger.error(f"音频匿名化失败: {e}")
            return None
    
    def transcribe_only(
        self,
        audio_path: Union[str, Path],
        **kwargs
    ) -> AlignedTranscriptionResult:
        """
        仅执行语音转录
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 转录参数
            
        Returns:
            转录结果
        """
        return self._transcribe_audio(Path(audio_path), **kwargs)
    
    def diarize_only(
        self,
        audio_path: Union[str, Path],
        **kwargs
    ) -> DiarizationResult:
        """
        仅执行说话人分离
        
        Args:
            audio_path: 音频文件路径
            **kwargs: 分离参数
            
        Returns:
            说话人分离结果
        """
        return self._diarize_audio(Path(audio_path), **kwargs)
    
    def detect_pii_only(
        self,
        transcription_result: AlignedTranscriptionResult,
        diarization_result: Optional[DiarizationResult] = None,
        **kwargs
    ) -> AudioPIIResult:
        """
        仅执行PII检测 - 简化版
        
        Args:
            transcription_result: 转录结果
            diarization_result: 说话人分离结果（可选）
            **kwargs: 检测参数
            
        Returns:
            PII检测结果
        """
        return self._detect_pii(transcription_result, diarization_result, **kwargs)
    
    def anonymize_audio_with_beep(
        self,
        audio_path: Union[str, Path],
        pii_result: AudioPIIResult,
        output_path: Optional[Union[str, Path]] = None
    ) -> Optional[str]:
        """使用蜂鸣音匿名化音频"""
        if not self._voice_anonymizer:
            return None
            
        try:
            return self._voice_anonymizer.anonymize_audio(
                audio_path=str(audio_path),
                pii_result=pii_result, 
                output_path=str(output_path) if output_path else None
            )
        except Exception as e:
            self.logger.error(f"蜂鸣音匿名化失败: {e}")
            return None