#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频工具模块

提供音频处理相关的工具函数和辅助功能。
"""

import os
import logging
import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, Union, Dict, Any
from pathlib import Path

from src.commons.loggers import get_module_logger


class AudioUtils:
    """音频工具类"""
    
    def __init__(self):
        self.logger = get_module_logger(__name__)
    
    @staticmethod
    def load_audio(
        file_path: Union[str, Path],
        sr: Optional[int] = None,
        mono: bool = True,
        offset: float = 0.0,
        duration: Optional[float] = None
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件
        
        Args:
            file_path: 音频文件路径
            sr: 目标采样率，None表示保持原始采样率
            mono: 是否转换为单声道
            offset: 开始时间偏移（秒）
            duration: 持续时间（秒），None表示加载全部
            
        Returns:
            音频数据和采样率的元组
        """
        try:
            audio, sample_rate = librosa.load(
                file_path,
                sr=sr,
                mono=mono,
                offset=offset,
                duration=duration
            )
            return audio, sample_rate
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"加载音频文件失败 {file_path}: {e}")
            raise
    
    @staticmethod
    def save_audio(
        audio: np.ndarray,
        file_path: Union[str, Path],
        sr: int,
        format: str = 'wav'
    ) -> None:
        """
        保存音频文件
        
        Args:
            audio: 音频数据
            file_path: 输出文件路径
            sr: 采样率
            format: 音频格式
        """
        try:
            # 确保输出目录存在
            output_dir = Path(file_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sf.write(file_path, audio, sr, format=format)
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"保存音频文件失败 {file_path}: {e}")
            raise
    
    @staticmethod
    def get_audio_info(file_path: Union[str, Path]) -> Dict[str, Any]:
        """
        获取音频文件信息
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            音频信息字典
        """
        try:
            info = sf.info(file_path)
            return {
                'duration': info.duration,
                'sample_rate': info.samplerate,
                'channels': info.channels,
                'frames': info.frames,
                'format': info.format,
                'subtype': info.subtype
            }
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"获取音频信息失败 {file_path}: {e}")
            raise
    
    @staticmethod
    def validate_audio_file(file_path: Union[str, Path]) -> bool:
        """
        验证音频文件是否有效
        
        Args:
            file_path: 音频文件路径
            
        Returns:
            文件是否有效
        """
        try:
            file_path = Path(file_path)
            
            # 检查文件是否存在
            if not file_path.exists():
                return False
            
            # 检查文件扩展名
            valid_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.mp4'}
            if file_path.suffix.lower() not in valid_extensions:
                return False
            
            # 尝试读取音频信息
            sf.info(file_path)
            return True
            
        except Exception:
            return False
    
    @staticmethod
    def normalize_audio(audio: np.ndarray, target_level: float = -20.0) -> np.ndarray:
        """
        音频电平归一化
        
        Args:
            audio: 音频数据
            target_level: 目标电平（dB）
            
        Returns:
            归一化后的音频数据
        """
        try:
            # 计算当前RMS
            rms = np.sqrt(np.mean(audio ** 2))
            
            if rms == 0:
                return audio
            
            # 计算目标RMS
            target_rms = 10 ** (target_level / 20)
            
            # 计算缩放因子
            scale_factor = target_rms / rms
            
            # 应用缩放并防止裁剪
            normalized_audio = audio * scale_factor
            max_val = np.max(np.abs(normalized_audio))
            
            if max_val > 1.0:
                normalized_audio = normalized_audio / max_val * 0.95
            
            return normalized_audio
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"音频归一化失败: {e}")
            return audio
    
    @staticmethod
    def resample_audio(
        audio: np.ndarray,
        orig_sr: int,
        target_sr: int
    ) -> np.ndarray:
        """
        重新采样音频
        
        Args:
            audio: 音频数据
            orig_sr: 原始采样率
            target_sr: 目标采样率
            
        Returns:
            重采样后的音频数据
        """
        try:
            if orig_sr == target_sr:
                return audio
            
            return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"音频重采样失败: {e}")
            return audio
    
    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sr: int,
        top_db: int = 20,
        frame_length: int = 2048,
        hop_length: int = 512
    ) -> np.ndarray:
        """
        修剪音频开头和结尾的静音
        
        Args:
            audio: 音频数据
            sr: 采样率
            top_db: 静音阈值（dB）
            frame_length: 帧长度
            hop_length: 跳跃长度
            
        Returns:
            修剪后的音频数据
        """
        try:
            trimmed_audio, _ = librosa.effects.trim(
                audio,
                top_db=top_db,
                frame_length=frame_length,
                hop_length=hop_length
            )
            return trimmed_audio
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"音频修剪失败: {e}")
            return audio
    
    @staticmethod
    def split_audio_by_time(
        audio: np.ndarray,
        sr: int,
        start_time: float,
        end_time: float
    ) -> np.ndarray:
        """
        按时间分割音频
        
        Args:
            audio: 音频数据
            sr: 采样率
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            
        Returns:
            分割后的音频数据
        """
        try:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 确保索引在有效范围内
            start_sample = max(0, start_sample)
            end_sample = min(len(audio), end_sample)
            
            if start_sample >= end_sample:
                return np.array([])
            
            return audio[start_sample:end_sample]
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"音频分割失败: {e}")
            return audio
    
    @staticmethod
    def calculate_audio_stats(audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """
        计算音频统计信息
        
        Args:
            audio: 音频数据
            sr: 采样率
            
        Returns:
            音频统计信息字典
        """
        try:
            stats = {
                'duration': len(audio) / sr,
                'sample_rate': sr,
                'samples': len(audio),
                'rms': np.sqrt(np.mean(audio ** 2)),
                'max_amplitude': np.max(np.abs(audio)),
                'zero_crossing_rate': np.mean(librosa.feature.zero_crossing_rate(audio)),
                'spectral_centroid': np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr)),
                'spectral_rolloff': np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
            }
            
            # 计算频谱特征
            if len(audio) > 0:
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
                stats['mfcc_mean'] = np.mean(mfccs, axis=1).tolist()
                stats['mfcc_std'] = np.std(mfccs, axis=1).tolist()
            
            return stats
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"计算音频统计信息失败: {e}")
            return {}
    
    @staticmethod
    def convert_to_mono(audio: np.ndarray) -> np.ndarray:
        """
        将立体声音频转换为单声道
        
        Args:
            audio: 音频数据（可能是多声道）
            
        Returns:
            单声道音频数据
        """
        try:
            if audio.ndim == 1:
                return audio
            elif audio.ndim == 2:
                # 如果是立体声，取平均值
                return np.mean(audio, axis=0)
            else:
                # 多声道，取第一个声道
                return audio[0]
                
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"转换单声道失败: {e}")
            return audio
    
    @staticmethod
    def apply_fade(
        audio: np.ndarray,
        sr: int,
        fade_in_duration: float = 0.1,
        fade_out_duration: float = 0.1
    ) -> np.ndarray:
        """
        应用淡入淡出效果
        
        Args:
            audio: 音频数据
            sr: 采样率
            fade_in_duration: 淡入时间（秒）
            fade_out_duration: 淡出时间（秒）
            
        Returns:
            应用淡入淡出后的音频数据
        """
        try:
            fade_in_samples = int(fade_in_duration * sr)
            fade_out_samples = int(fade_out_duration * sr)
            
            audio_with_fade = audio.copy()
            
            # 应用淡入
            if fade_in_samples > 0 and len(audio) > fade_in_samples:
                fade_in_curve = np.linspace(0, 1, fade_in_samples)
                audio_with_fade[:fade_in_samples] *= fade_in_curve
            
            # 应用淡出
            if fade_out_samples > 0 and len(audio) > fade_out_samples:
                fade_out_curve = np.linspace(1, 0, fade_out_samples)
                audio_with_fade[-fade_out_samples:] *= fade_out_curve
            
            return audio_with_fade
            
        except Exception as e:
            logger = get_module_logger(__name__)
            logger.error(f"应用淡入淡出失败: {e}")
            return audio


# 便捷函数
def load_audio_file(file_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, int]:
    """
    加载音频文件的便捷函数
    
    Args:
        file_path: 音频文件路径
        **kwargs: 传递给 AudioUtils.load_audio 的参数
        
    Returns:
        音频数据和采样率的元组
    """
    return AudioUtils.load_audio(file_path, **kwargs)


def save_audio_file(audio: np.ndarray, file_path: Union[str, Path], sr: int, **kwargs) -> None:
    """
    保存音频文件的便捷函数
    
    Args:
        audio: 音频数据
        file_path: 输出文件路径
        sr: 采样率
        **kwargs: 传递给 AudioUtils.save_audio 的参数
    """
    AudioUtils.save_audio(audio, file_path, sr, **kwargs)


def validate_audio(file_path: Union[str, Path]) -> bool:
    """
    验证音频文件的便捷函数
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        文件是否有效
    """
    return AudioUtils.validate_audio_file(file_path)


def get_audio_duration(file_path: Union[str, Path]) -> float:
    """
    获取音频文件时长的便捷函数
    
    Args:
        file_path: 音频文件路径
        
    Returns:
        音频时长（秒）
    """
    try:
        info = AudioUtils.get_audio_info(file_path)
        return info.get('duration', 0.0)
    except Exception:
        return 0.0
