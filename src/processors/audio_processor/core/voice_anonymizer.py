#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
语音匿名化模块 - 简化版

根据 AudioPIIResult 对音频中的PII位置进行蜂鸣声处理。
"""

import numpy as np
import librosa
import soundfile as sf
from typing import Optional
from pathlib import Path

from src.commons.loggers import get_module_logger
from src.configs.processors.audio_processor.voice_anonymizer_config import VoiceAnonymizerConfig
from .pii_detection import AudioPIIResult

# 获取模块日志记录器
logger = get_module_logger(__name__)


class VoiceAnonymizer:
    """语音匿名化处理器 - 简化版
    
    根据 AudioPIIResult 对音频中的PII位置进行蜂鸣声替换。
    """
    
    def __init__(self, config: VoiceAnonymizerConfig):
        """初始化语音匿名化器
        
        Args:
            config: 语音匿名化配置
        """
        self.config = config
        
        logger.info(f"初始化语音匿名化器 - 采样率: {self.config.sample_rate}")
    
    def anonymize_audio(self, 
                       audio_path: str, 
                       pii_result: AudioPIIResult,
                       output_path: Optional[str] = None) -> str:
        """对音频进行匿名化处理
        
        Args:
            audio_path: 输入音频文件路径
            pii_result: PII检测结果
            output_path: 输出音频文件路径，如果为None则自动生成
        
        Returns:
            str: 处理后的音频文件路径
        """
        try:
            # 加载音频
            audio, sr = librosa.load(audio_path, sr=self.config.sample_rate)
            
            # 如果没有检测到PII，直接返回原音频
            if not pii_result.pii_entities:
                logger.info("未检测到PII实体，返回原始音频")
                if output_path and output_path != audio_path:
                    sf.write(output_path, audio, sr)
                    return output_path
                return audio_path
            
            # 复制音频数据进行处理
            processed_audio = audio.copy()
            
            # 处理每个PII实体
            for entity in pii_result.pii_entities:
                if entity.start_time is not None and entity.end_time is not None:
                    self._apply_beep(processed_audio, entity.start_time, entity.end_time, int(sr))
            
            # 生成输出路径
            if output_path is None:
                base_name = Path(audio_path).stem
                output_path = str(Path(audio_path).parent / f"{base_name}_anonymized.wav")
            
            # 保存处理后的音频
            sf.write(output_path, processed_audio, sr)
            
            logger.info(f"音频匿名化完成，处理了 {len(pii_result.pii_entities)} 个PII实体，输出: {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"音频匿名化失败: {e}")
            raise
    
    def _apply_beep(self, audio: np.ndarray, start_time: float, end_time: float, sample_rate: int) -> None:
        """在指定时间段应用蜂鸣音（就地修改音频）
        
        Args:
            audio: 音频数组
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            sample_rate: 采样率
        """
        # 计算样本索引
        start_sample = int(start_time * sample_rate)
        end_sample = int(end_time * sample_rate)
        
        # 确保索引在有效范围内
        start_sample = max(0, start_sample)
        end_sample = min(len(audio), end_sample)
        
        if start_sample >= end_sample:
            return
        
        # 生成蜂鸣音
        duration = (end_sample - start_sample) / sample_rate
        beep = self._generate_beep(duration, sample_rate)
        
        # 应用淡入淡出
        if self.config.fade_duration > 0:
            beep = self._apply_fade(beep, sample_rate)
        
        # 替换音频片段
        segment_length = end_sample - start_sample
        if len(beep) > segment_length:
            beep = beep[:segment_length]
        elif len(beep) < segment_length:
            # 如果蜂鸣音太短，重复或填充
            beep = np.tile(beep, int(np.ceil(segment_length / len(beep))))[:segment_length]
        
        audio[start_sample:end_sample] = beep
    
    def _generate_beep(self, duration: float, sample_rate: int) -> np.ndarray:
        """生成蜂鸣音
        
        Args:
            duration: 持续时间（秒）
            sample_rate: 采样率
            
        Returns:
            蜂鸣音数组
        """
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        beep = self.config.beep_amplitude * np.sin(2 * np.pi * self.config.beep_frequency * t)
        return beep.astype(np.float32)
    
    def _apply_fade(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """应用淡入淡出效果
        
        Args:
            audio: 音频数组
            sample_rate: 采样率
            
        Returns:
            应用淡入淡出后的音频数组
        """
        fade_samples = int(self.config.fade_duration * sample_rate)
        fade_samples = min(fade_samples, len(audio) // 2)
        
        if fade_samples <= 0:
            return audio
        
        # 复制音频以避免就地修改参数
        audio = audio.copy()
        
        # 应用淡入
        fade_in = np.linspace(0, 1, fade_samples)
        audio[:fade_samples] *= fade_in
        
        # 应用淡出
        fade_out = np.linspace(1, 0, fade_samples)
        audio[-fade_samples:] *= fade_out
        
        return audio