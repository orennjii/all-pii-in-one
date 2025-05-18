"""
音频匿名化处理器
集成说话人分割、语音转换和PII检测功能
"""

import os
import tempfile
import shutil
import uuid
import torch
import numpy as np
import librosa
import soundfile as sf
from typing import Dict, List, Tuple, Optional, Union, Any
from src.commons.device_config import get_device
from src.commons.utils import create_temp_dir, cleanup_temp_dir, ensure_directory_exists

from .speaker_diarization import SpeakerDiarization
from .voice_conversion import VoiceConverter
from .audio_pii_detector import AudioPIIDetector, SpeechToTextPIIDetector


class AudioAnonymizer:
    """
    音频匿名化器
    
    功能: 
    1. 说话人分割: 识别音频中的不同说话人
    2. 语音转换: 将识别出的说话人声音替换为匿名声音
    3. PII检测: 可选功能, 检测音频中的个人隐私信息
    
    处理流程: 
    1. 输入音频 -> 说话人分割 -> 提取说话人片段
    2. 每个说话人片段 -> 语音转换 -> 匿名化声音
    3. 合并所有匿名化片段 -> 输出匿名化音频
    4. (可选) PII检测 -> 标记包含PII的片段
    """

    def __init__(
        self,
        device: str = "cpu",
        reference_voices_dir: Optional[str] = None,
        enable_pii_detection: bool = False,
        pii_detector: Optional[AudioPIIDetector] = None
    ):
        """
        初始化音频匿名化器
        
        Args:
            device: 运行设备, "cpu"或"cuda"
            hf_access_token: HuggingFace访问令牌, 用于下载模型
            diarization_model_path: 说话人分割模型路径
            voice_models_dir: 语音转换模型目录
            reference_voices_dir: 参考声音目录
            enable_pii_detection: 是否启用PII检测
            pii_detector: 自定义PII检测器
        """
        self.device = device
        self.enable_pii_detection = enable_pii_detection
        self.reference_voices_dir = reference_voices_dir or os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 
            "../data/reference_voices"
        )
        
        # 初始化说话人分割器
        self.diarizer = SpeakerDiarization(
            device=device
        )
        
        # 初始化语音转换器
        self.voice_converter = VoiceConverter(
            device=device,
        )
        
        # 初始化PII检测器
        if enable_pii_detection:
            self.pii_detector = pii_detector or SpeechToTextPIIDetector()
        else:
            self.pii_detector = None
        
        # 获取参考声音列表
        self.reference_voices = self._get_reference_voices()
    
    def _get_reference_voices(self) -> Dict[str, str]:
        """
        获取参考声音列表
        
        Returns:
            Dict[str, str]: 参考声音名称到文件路径的映射
        """
        if not os.path.exists(self.reference_voices_dir):
            os.makedirs(self.reference_voices_dir, exist_ok=True)
            return {}
        
        reference_voices = {}
        for file in os.listdir(self.reference_voices_dir):
            if file.lower().endswith('.wav'):
                name = os.path.splitext(file)[0]
                path = os.path.join(self.reference_voices_dir, file)
                reference_voices[name] = path
        
        return reference_voices
    
    def add_reference_voice(self, voice_path: str, name: Optional[str] = None) -> str:
        """
        添加参考声音
        
        Args:
            voice_path: 参考声音文件路径
            name: 参考声音名称, 如果为None则使用文件名
            
        Returns:
            str: 参考声音在系统中的ID
        """
        if not os.path.exists(voice_path):
            raise FileNotFoundError(f"参考声音文件不存在: {voice_path}")
        
        # 如果未提供名称, 使用文件名
        if name is None:
            name = os.path.splitext(os.path.basename(voice_path))[0]
        
        # 确保名称唯一
        if name in self.reference_voices:
            name = f"{name}_{str(uuid.uuid4())[:8]}"
        
        # 复制到参考声音目录
        target_path = os.path.join(self.reference_voices_dir, f"{name}.wav")
        shutil.copy(voice_path, target_path)
        
        # 更新参考声音列表
        self.reference_voices[name] = target_path
        
        return name
    
    def anonymize(self, 
                 audio_path: str, 
                 output_path: Optional[str] = None,
                 reference_voice: Optional[str] = None,
                 min_speakers: int = 1,
                 max_speakers: int = 5,
                 min_segment_duration: float = 1.0,
                 keep_original_segments: bool = False,
                 return_pii_info: bool = False,
                 diffusion_steps: Optional[int] = None,
                 length_adjust: Optional[float] = None,
                 speaker_mapping: Optional[Dict[str, str]] = None,
                 **kwargs) -> Union[str, Tuple[str, Dict]]:
        """
        匿名化音频
        
        Args:
            audio_path: 输入音频文件路径
            output_path: 输出音频文件路径, 如果为None则自动生成
            reference_voice: 参考声音名称或文件路径, 如果为None则从可用的参考声音中随机选择
            min_speakers: 最少说话人数
            max_speakers: 最多说话人数
            min_segment_duration: 最小片段时长（秒）
            keep_original_segments: 是否保留原始音频片段
            return_pii_info: 是否返回PII检测信息
            diffusion_steps: 扩散步数, 值越大质量越高但处理时间越长
            length_adjust: 语音速度调整, <1.0加速, >1.0减慢
            speaker_mapping: 说话人ID到参考声音的映射, 用于为不同说话人分配不同的参考声音
            **kwargs: 传递给语音转换器的额外参数
            
        Returns:
            Union[str, Tuple[str, Dict]]: 
                - 如果return_pii_info为False, 返回匿名化音频路径
                - 如果return_pii_info为True, 返回(匿名化音频路径, PII检测结果)
        """
        # 创建临时目录
        temp_dir = create_temp_dir(prefix="audio_anonymizer_")
        
        try:
            print(f"正在处理音频: {audio_path}")
            
            # 步骤1: 说话人分割 - 使用SpeakerDiarization的__call__方法一次完成分割和片段提取
            print(f"正在进行说话人分割...")
            segments_dir = os.path.join(temp_dir, "segments")
            diarization_results = self.diarizer(
                audio_file=audio_path,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                min_segment_duration=min_segment_duration,
                output_dir=segments_dir
            )
            
            segments = diarization_results['segments']
            speaker_audio_paths = diarization_results.get('speaker_audio_paths', {})
            
            # 获取检测到的说话人数量
            num_speakers = len(speaker_audio_paths)
            print(f"检测到 {num_speakers} 个说话人, 共 {len(segments)} 个语音片段")
            
            # 如果没有检测到说话人，直接返回原始音频
            if not segments:
                print("未检测到有效的说话人片段, 返回原始音频")
                if output_path is None:
                    output_dir = os.path.dirname(audio_path)
                    output_name = f"anonymized_{os.path.basename(audio_path)}"
                    output_path = os.path.join(output_dir, output_name)
                shutil.copy(audio_path, output_path)
                return output_path
            
            # 步骤2: 准备参考声音
            print(f"准备参考声音...")
            # 如果未指定speaker_mapping，则创建一个
            if speaker_mapping is None:
                speaker_mapping = {}
            
            # 为每个说话人分配参考声音
            for speaker_id in speaker_audio_paths.keys():
                if speaker_id not in speaker_mapping:
                    # 如果没有为该说话人指定参考声音
                    if reference_voice is None:
                        # 如果没有指定默认参考声音，随机选择一个
                        if not self.reference_voices:
                            raise ValueError("没有可用的参考声音, 请先添加参考声音")
                        selected_voice = list(self.reference_voices.values())[
                            hash(speaker_id) % len(self.reference_voices)
                        ]
                    elif reference_voice in self.reference_voices:
                        # 使用指定的参考声音名称
                        selected_voice = self.reference_voices[reference_voice]
                    elif os.path.exists(reference_voice):
                        # 使用指定的参考声音文件路径
                        selected_voice = reference_voice
                    else:
                        raise FileNotFoundError(f"参考声音文件不存在: {reference_voice}")
                    
                    # 添加到映射
                    speaker_mapping[speaker_id] = selected_voice
            
            # 步骤3: 语音转换 - 为每个说话人的所有片段进行批量转换
            print(f"正在进行语音转换...")
            converted_dir = os.path.join(temp_dir, "converted")
            ensure_directory_exists(converted_dir)
            
            converted_segments = []
            for speaker_id, audio_paths in speaker_audio_paths.items():
                # 获取此说话人的参考声音
                speaker_reference = speaker_mapping[speaker_id]
                print(f"为说话人 {speaker_id} 转换 {len(audio_paths)} 个片段, 使用参考声音: {os.path.basename(speaker_reference)}")
                
                # 确保输出目录存在
                speaker_output_dir = os.path.join(converted_dir, f"speaker_{speaker_id}")
                ensure_directory_exists(speaker_output_dir)
                
                # 批量转换该说话人的所有片段
                converted_paths = self.voice_converter.batch_convert(
                    source_paths=audio_paths,
                    target_dir=speaker_output_dir,
                    reference_voice=speaker_reference,
                    diffusion_steps=diffusion_steps,
                    length_adjust=length_adjust,
                    **kwargs
                )
                
                # 将转换后的路径与原始片段信息关联
                for i, path in enumerate(converted_paths):
                    if i < len(audio_paths):
                        # 查找该音频文件对应的原始片段信息
                        original_path = audio_paths[i]
                        segment_filename = os.path.basename(original_path)
                        # 从文件名中提取片段索引
                        segment_index = int(segment_filename.split('_')[1].split('.')[0])
                        
                        if segment_index < len(segments):
                            # 找到匹配的片段信息
                            matching_segments = [
                                s for s in segments 
                                if s['speaker'] == speaker_id and 
                                   abs(segments[segment_index]['start'] - s['start']) < 0.01
                            ]
                            
                            if matching_segments:
                                segment_info = matching_segments[0].copy()
                                segment_info['converted_path'] = path
                                segment_info['original_path'] = original_path
                                converted_segments.append(segment_info)
            
            # 确保片段按时间排序
            converted_segments.sort(key=lambda x: x['start'])
            
            print(f"成功转换 {len(converted_segments)} 个语音片段")
            
            # 步骤4: 合并所有转换后的片段
            if output_path is None:
                # 如果未指定输出路径, 生成一个
                output_dir = os.path.dirname(audio_path)
                output_name = f"anonymized_{os.path.basename(audio_path)}"
                output_path = os.path.join(output_dir, output_name)
            
            # 确保输出目录存在
            ensure_directory_exists(os.path.dirname(os.path.abspath(output_path)))
            
            # 合并音频片段
            print(f"正在合并所有转换后的片段...")
            self._merge_audio_segments(
                segments=converted_segments,
                original_audio_path=audio_path,
                output_path=output_path,
                keep_original=keep_original_segments
            )
            
            print(f"已完成音频匿名化, 结果保存至: {output_path}")
            
            # 步骤5: (可选) PII检测
            pii_result = None
            if self.enable_pii_detection and self.pii_detector and return_pii_info:
                print("正在检测音频中的PII信息...")
                pii_result = self.pii_detector.detect(output_path)
                if pii_result.get('pii_found', False):
                    print(f"在音频中检测到 {len(pii_result.get('segments', []))} 个PII片段")
                else:
                    print("未在音频中检测到PII信息")
            
            # 返回结果
            if return_pii_info:
                return output_path, pii_result
            else:
                return output_path
                
        finally:
            # 清理临时文件
            cleanup_temp_dir(temp_dir)

    def _merge_audio_segments(self, 
                             segments: List[Dict], 
                             original_audio_path: str, 
                             output_path: str,
                             keep_original: bool = False) -> str:
        """
        合并音频片段
        
        Args:
            segments: 片段列表, 包含start、end和converted_path
            original_audio_path: 原始音频路径, 用于获取背景噪音和采样率
            output_path: 输出音频路径
            keep_original: 是否保留未转换的片段
            
        Returns:
            str: 合并后的音频文件路径
        """
        # 加载原始音频, 用于获取采样率和总长度
        y_orig, sr = librosa.load(original_audio_path, sr=None)
        total_duration = librosa.get_duration(y=y_orig, sr=sr)
        
        # 创建空白音频, 与原始音频长度相同
        y_output = np.zeros_like(y_orig)
        
        # 遍历所有片段
        for segment in segments:
            start_time = segment['start']
            end_time = segment['end']
            
            # 计算对应的采样点
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            # 载入转换后的音频片段
            if keep_original:
                y_segment, segment_sr = librosa.load(segment['original_path'], sr=sr)
            else:
                y_segment, segment_sr = librosa.load(segment['converted_path'], sr=sr)
            
            # 如果片段长度与预期不符, 进行调整
            expected_length = end_sample - start_sample
            actual_length = len(y_segment)
            
            if actual_length > expected_length:
                # 截断
                y_segment = y_segment[:expected_length]
            elif actual_length < expected_length:
                # 填充
                padding = np.zeros(expected_length - actual_length)
                y_segment = np.concatenate([y_segment, padding])
            
            # 将片段插入到输出音频中
            y_output[start_sample:end_sample] = y_segment
        
        # 保存合并后的音频
        sf.write(output_path, y_output, sr)
        
        return output_path
    
    def detect_pii(self, audio_path: str) -> Dict[str, Any]:
        """
        检测音频中的PII信息
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            Dict: PII检测结果
        """
        if not self.enable_pii_detection or not self.pii_detector:
            raise ValueError("未启用PII检测, 请先启用PII检测功能")
        
        return self.pii_detector.detect(audio_path)
