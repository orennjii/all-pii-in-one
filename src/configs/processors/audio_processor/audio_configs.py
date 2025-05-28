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

class AudioDiarizationConfig(BaseConfig):
    """说话人分割配置"""
    
    enabled: bool = Field(
        default=True,
        description="是否启用说话人分割"
    )
    auth_token: str = Field(
        default="",
        description="Hugging Face认证令牌，用于访问私有模型"
    )
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


class AudioTranscriptionConfig(BaseConfig):
    """语音转录配置"""
    
    # 模型配置
    model_size: str = Field(
        default="base",
        description="模型大小: tiny, base, small, medium, large-v1, large-v2, large-v3"
    )
    language: Optional[str] = Field(
        default=None,
        description="目标语言，None表示自动检测"
    )
    compute_type: str = Field(
        default="float16",
        description="计算精度: float16, float32, int8"
    )
    device: Optional[str] = Field(
        default=None,
        description="计算设备: cuda, cpu, auto"
    )
    
    # WhisperX 特定配置
    batch_size: int = Field(
        default=16,
        description="批处理大小"
    )
    chunk_size: int = Field(
        default=30,
        description="音频块大小（秒）"
    )
    
    # 对齐配置
    align_model: Optional[str] = Field(
        default=None,
        description="对齐模型名称，None表示自动选择"
    )
    interpolate_method: str = Field(
        default="nearest",
        description="插值方法"
    )
    return_char_alignments: bool = Field(
        default=False,
        description="是否返回字符级对齐"
    )
    
    # 语音活动检测配置
    vad_onset: float = Field(
        default=0.500,
        description="VAD起始阈值"
    )
    vad_offset: float = Field(
        default=0.363,
        description="VAD结束阈值"
    )
    vad_min_duration_on: float = Field(
        default=0.0,
        description="最小激活时长"
    )
    vad_min_duration_off: float = Field(
        default=0.0,
        description="最小静默时长"
    )
    
    # 解码配置
    temperature: float = Field(
        default=0.0,
        description="采样温度"
    )
    best_of: Optional[int] = Field(
        default=None,
        description="候选数量"
    )
    beam_size: Optional[int] = Field(
        default=None,
        description="束搜索大小"
    )
    patience: Optional[float] = Field(
        default=None,
        description="束搜索耐心值"
    )
    length_penalty: Optional[float] = Field(
        default=None,
        description="长度惩罚"
    )
    suppress_tokens: str = Field(
        default="-1",
        description="抑制的令牌"
    )
    initial_prompt: Optional[str] = Field(
        default=None,
        description="初始提示"
    )
    condition_on_previous_text: bool = Field(
        default=True,
        description="是否基于前文条件化"
    )
    fp16: bool = Field(
        default=True,
        description="是否使用半精度"
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        description="压缩率阈值"
    )
    logprob_threshold: float = Field(
        default=-1.0,
        description="对数概率阈值"
    )
    no_speech_threshold: float = Field(
        default=0.6,
        description="无语音阈值"
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