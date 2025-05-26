#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理器配置模块
提供音频处理所需的配置项和默认值
"""

import os
from pathlib import Path
from typing import List, Optional
from pydantic import Field

from src.commons import find_project_root
from src.configs.base_config import BaseConfig


class AudioSupportedFormatsConfig(BaseConfig):
    """支持的音频格式配置"""
    
    file_types: List[str] = Field(
        default=[".wav", ".mp3", ".flac", ".ogg"],
        description="支持的音频文件类型扩展名"
    )


class AudioDiarizationConfig(BaseConfig):
    """说话人分割配置"""
    
    model: str = Field(
        default="pyannote/speaker-diarization-3.1",
        description="使用的说话人分割模型"
    )
    min_speakers: int = Field(
        default=1,
        description="最小说话人数量"
    )
    max_speakers: int = Field(
        default=5,
        description="最大说话人数量"
    )
    min_segment_duration: float = Field(
        default=1.0,
        description="最小分段时长（秒）"
    )
    segmentation_onset: Optional[float] = Field(
        default=None,
        description="语音活动检测（VAD）的起始阈值"
    )
    min_duration_on: Optional[float] = Field(
        default=None,
        description="用于移除过短的语音片段"
    )
    min_duration_off: Optional[float] = Field(
        default=None,
        description="用于填充过短的非语音间隙"
    )


class AudioVoiceConversionConfig(BaseConfig):
    """语音转换配置"""
    
    diffusion_steps: int = Field(
        default=10,
        description="扩散模型步数，影响生成质量和速度"
    )
    length_adjust: float = Field(
        default=1.0,
        description="长度调整因子，<1.0加速语速，>1.0减慢语速"
    )
    inference_cfg_rate: float = Field(
        default=0.7,
        description="推理CFG速率，影响输出音质"
    )
    f0_condition: bool = Field(
        default=True,
        description="是否使用F0条件，保留音高信息"
    )
    auto_f0_adjust: bool = Field(
        default=True,
        description="是否自动调整F0以匹配参考声音"
    )
    default_pitch_shift: int = Field(
        default=0,
        description="默认音高调整值（半音）"
    )


class AudioProcessorConfig(BaseConfig):
    """音频处理器的总体配置"""
    
    supported_formats: AudioSupportedFormatsConfig = Field(
        default_factory=AudioSupportedFormatsConfig,
        description="支持的音频格式配置"
    )
    diarization: AudioDiarizationConfig = Field(
        default_factory=AudioDiarizationConfig,
        description="说话人分割配置"
    )
    voice_conversion: AudioVoiceConversionConfig = Field(
        default_factory=AudioVoiceConversionConfig,
        description="语音转换配置"
    )
    enable_pii_detection: bool = Field(
        default=False,
        description="是否启用PII检测"
    )
    reference_voices_dir: Optional[str] = Field(
        default=None,
        description="参考声音目录路径，如果为None将使用默认路径"
    )
    
    def get_reference_voices_dir(self) -> str:
        """获取参考声音目录路径，如果未配置则返回默认路径"""
        if self.reference_voices_dir:
            return self.reference_voices_dir
        
        # 使用默认路径：项目根目录/data/audio/reference_voices
        project_root = find_project_root(Path(__file__))
        return os.path.join(project_root, "data", "audio", "reference_voices")


__all__ = [
    "AudioSupportedFormatsConfig",
    "AudioDiarizationConfig", 
    "AudioVoiceConversionConfig",
    "AudioProcessorConfig"
]
