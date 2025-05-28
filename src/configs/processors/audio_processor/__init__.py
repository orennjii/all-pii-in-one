#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理器配置模块
"""

from typing import List
from pydantic import Field
from src.configs.base_config import BaseConfig
from src.configs.processors.audio_processor.audio_configs import (
    AudioDiarizationConfig,
    AudioVoiceConversionConfig,
    AudioTranscriptionConfig
)
from src.configs.processors.audio_processor.voice_anonymizer_config import VoiceAnonymizerConfig

class AudioProcessorConfig(BaseConfig):
    """音频处理器配置"""

    supported_formats: List[str] = Field(
        default=[".wav", ".mp3", ".flac", ".ogg"],
        description="支持的音频文件类型扩展名"
    )

    enable_pii_detection: bool = Field(
        default=False,
        description="是否启用PII（个人身份信息）检测"
    )

    reference_audio_path: str = Field(
        default="data/audio/reference_voice",
        description="参考音频文件相对路径，用于音频处理"
    )
    
    diarization: AudioDiarizationConfig = Field(
        default_factory=AudioDiarizationConfig,
        description="说话人分割配置"
    )

    transcription: AudioTranscriptionConfig = Field(
        default_factory=AudioTranscriptionConfig,
        description="语音转录配置"
    )

    voice_conversion: AudioVoiceConversionConfig = Field(
        default_factory=AudioVoiceConversionConfig,
        description="语音转换配置"
    )

    voice_anonymizer: VoiceAnonymizerConfig = Field(
        default_factory=VoiceAnonymizerConfig,
        description="语音匿名化配置"
    )

