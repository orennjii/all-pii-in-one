"""
说话人分割模块, 使用Pyannote Audio进行处理.

此模块提供音频中的说话人分割功能, 可以识别音频中不同的说话人并提取相应的音频片段.
基于Pyannote Audio库实现, 支持自定义模型和设备.
"""

# 标准库
import os
from typing import Dict, List, Optional

# 第三方库
import librosa
import soundfile as sf
import torch
from pyannote.audio import Pipeline

# 内部模块
# (无内部模块导入)


class SpeakerDiarization:
    """
    使用Pyannote Audio进行说话人分割.
    
    此类提供音频中说话人识别和分割功能, 可以确定谁在何时说话.
    """
    
    def __init__(
        self,
        device: torch.device,
    ) -> None:
        """
        初始化说话人分割器.
        
        Args:
            device: 计算设备 (torch.device).
        """
        self._device = device
        self._pipeline: Pipeline = None

    def _create_pipeline(
            self, 
            model_name_or_path: str = "pyannote/speaker-diarization-3.1", 
            use_local_model: bool = False,
            access_token: Optional[str] = None
        ) -> None:
        """根据传入参数创建并初始化Pyannote Pipeline.
        
        Args:
            model_name_or_path: 模型名称(HuggingFace Hub ID)或本地模型路径.
            use_local_model: 如果为True, 则model_name_or_path被视为本地路径;
                            否则视为HuggingFace模型ID.
            access_token: HuggingFace访问令牌, 用于下载HuggingFace模型.
        """
        if use_local_model:
            if not model_name_or_path or not os.path.exists(model_name_or_path):
                raise FileNotFoundError(
                    f"本地模型路径 '{model_name_or_path}' 未找到或未指定."
                )
            # 使用本地模型
            self._pipeline = Pipeline.from_pretrained(model_name_or_path)
        else:
            # 从HuggingFace下载模型
            self._pipeline = Pipeline.from_pretrained(
                model_name_or_path,
                use_auth_token=access_token
            )
        
        if not self._pipeline:
            raise RuntimeError("无法初始化Pyannote Pipeline.")

        self._pipeline.to(self._device)

    def _ensure_pipeline_initialized(
            self, 
            model_name_or_path: str = "pyannote/speaker-diarization-3.1", 
            use_local_model: bool = False,
            access_token: Optional[str] = None
        ) -> None:
        """确保Pipeline在使用前已初始化.
        
        Args:
            model_name_or_path: 模型名称(HuggingFace Hub ID)或本地模型路径.
            use_local_model: 如果为True, 则model_name_or_path被视为本地路径;
                            否则视为HuggingFace模型ID.
            access_token: HuggingFace访问令牌, 用于下载HuggingFace模型.
        """
        if self._pipeline is None:
            self._create_pipeline(
                model_name_or_path=model_name_or_path,
                use_local_model=use_local_model,
                access_token=access_token
            )

    def process(
            self, 
            audio_file: str, 
            min_speakers: int = 1, 
            max_speakers: int = 5,
            model_name_or_path: str = "pyannote/speaker-diarization-3.1",
            use_local_model: bool = False,
            access_token: Optional[str] = None
        ) -> Dict:
        """
        处理音频文件并进行说话人分割.
        
        此方法分析整个音频文件并识别其中的不同说话人,
        生成包含每个说话人说话时间段的分割结果.
        
        Args:
            audio_file: 音频文件路径.
            min_speakers: 最少说话人数.
            max_speakers: 最多说话人数.
            model_name_or_path: 模型名称(HuggingFace Hub ID)或本地模型路径.
            use_local_model: 如果为True, 则model_name_or_path被视为本地路径.
            access_token: HuggingFace访问令牌, 用于下载HuggingFace模型.
            
        Returns:
            Dict: 说话人分割结果, 包含说话人ID和对应的时间段.
        """
        self._ensure_pipeline_initialized(
            model_name_or_path=model_name_or_path,
            use_local_model=use_local_model,
            access_token=access_token
        )
            
        diarization = self._pipeline(
            audio_file, 
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        return diarization

    def get_speaker_segments(
            self, 
            diarization_result, 
            min_segment_duration: float = 1.0
        ) -> List[Dict]:
        """
        从分割结果中提取说话人片段.
        
        将分割结果转换为更易于处理的片段列表,
        每个片段包含说话人ID、开始和结束时间等信息.
        
        Args:
            diarization_result: Pyannote分割结果.
            min_segment_duration: 最小片段时长(秒), 小于此时长的片段将被过滤掉.
            
        Returns:
            List[Dict]: 说话人片段列表, 每个片段包含:
                - speaker: 说话人ID
                - start: 起始时间(秒)
                - end: 结束时间(秒)
                - duration: 持续时间(秒)
        """
        segments = []
        
        # 遍历所有说话人片段
        for turn, _, speaker in diarization_result.itertracks(yield_label=True):
            # 只处理长度超过阈值的片段
            if turn.duration >= min_segment_duration:
                segments.append({
                    'speaker': speaker,
                    'start': turn.start,
                    'end': turn.end,
                    'duration': turn.duration
                })
        
        return segments
    
    def extract_speaker_audio(
            self,
            audio_file: str,
            segments: List[Dict],
            output_dir: str
        ) -> Dict[str, List[str]]:
        """
        根据分割结果提取每个说话人的音频片段.
        
        从原始音频中截取每个说话人的语音片段, 并保存为单独的文件.
        
        Args:
            audio_file: 原始音频文件路径.
            segments: 说话人片段列表, 通常由get_speaker_segments方法生成.
            output_dir: 输出目录, 用于保存提取的音频片段.
            
        Returns:
            Dict[str, List[str]]: 每个说话人的音频片段路径列表,
                键为说话人ID, 值为该说话人的所有音频片段文件路径列表.
        """
        # 加载音频文件
        audio, sr = librosa.load(audio_file, sr=None)
        
        # 按说话人分组
        speaker_segments = {}
        for segment in segments:
            speaker = segment['speaker']
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            speaker_segments[speaker].append(segment)
        
        # 提取每个说话人的音频片段
        speaker_audio_paths = {}
        
        for speaker, segs in speaker_segments.items():
            speaker_dir = os.path.join(output_dir, f"speaker_{speaker}")
            os.makedirs(speaker_dir, exist_ok=True)
            
            speaker_audio_paths[speaker] = []
            
            for i, segment in enumerate(segs):
                start_sample = int(segment['start'] * sr)
                end_sample = int(segment['end'] * sr)
                
                # 提取音频片段
                segment_audio = audio[start_sample:end_sample]
                
                # 保存到文件
                output_path = os.path.join(speaker_dir, f"segment_{i:04d}.wav")
                sf.write(output_path, segment_audio, sr)
                
                speaker_audio_paths[speaker].append(output_path)
        
        return speaker_audio_paths

    def __call__(
        self,
        audio_file: str,
        min_speakers: int = 1,
        max_speakers: int = 5,
        min_segment_duration: float = 1.0,
        output_dir: Optional[str] = None,
        model_name_or_path: str = "pyannote/speaker-diarization-3.1",
        use_local_model: bool = False,
        access_token: Optional[str] = None
    ) -> Dict:
        """
        直接调用处理音频文件并返回所有结果.

        此方法是核心处理流程的封装, 便于直接调用.
        它会执行说话人分割, 提取片段信息, 并可选择提取每个说话人的音频.

        Args:
            audio_file: 音频文件路径.
            min_speakers: 最少说话人数.
            max_speakers: 最多说话人数.
            min_segment_duration: 最小片段时长(秒).
            output_dir: (可选) 输出目录, 用于保存提取的音频片段.
                        如果提供, 则会提取并保存音频.
            model_name_or_path: 模型名称(HuggingFace Hub ID)或本地模型路径.
            use_local_model: 如果为True, 则model_name_or_path被视为本地路径.
            access_token: HuggingFace访问令牌, 用于下载HuggingFace模型.

        Returns:
            Dict: 包含以下键值的结果字典:
                - 'diarization': Pyannote的原始分割结果.
                - 'segments': 说话人片段列表.
                - 'speaker_audio_paths': (可选) 每个说话人的音频片段路径列表.
                                         仅当提供了 output_dir 时包含此键.
        """
        results = {}

        # 1. 处理音频文件进行分割
        diarization_result = self.process(
            audio_file,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            model_name_or_path=model_name_or_path,
            use_local_model=use_local_model,
            access_token=access_token
        )
        results['diarization'] = diarization_result

        # 2. 获取说话人片段
        segments = self.get_speaker_segments(
            diarization_result,
            min_segment_duration=min_segment_duration
        )
        results['segments'] = segments

        # 3. (可选) 提取说话人音频
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)  # 确保输出目录存在
            speaker_audio_paths = self.extract_speaker_audio(
                audio_file,
                segments,
                output_dir
            )
            results['speaker_audio_paths'] = speaker_audio_paths
        
        return results

def main():
    """
    说话人分割模块测试函数.
    
    此函数提供SpeakerDiarization类的使用示例, 包括:
    1. 初始化说话人分割器
    2. 处理音频文件
    3. 获取并打印说话人片段
    4. 提取说话人音频(可选)
    """
    # 测试参数设置
    audio_file = "./data/audio/source_voices/source_s1.wav"  # 待处理音频文件路径
    output_dir = "./output/speaker_segments"  # 输出目录
    min_speakers = 1  # 最少说话人数
    max_speakers = 3  # 最多说话人数
    min_segment_duration = 1.0  # 最小片段时长(秒)
    access_token = None  # HuggingFace访问令牌, 如需使用请替换为实际token
    
    # 是否使用本地模型
    use_local_model = False
    model_name_or_path = "pyannote/speaker-diarization-3.1"  # 模型名称或本地路径
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    try:
        print("初始化说话人分割器...")
        # 初始化说话人分割器
        diarization = SpeakerDiarization(
            device=device
        )
        
        print(f"处理音频文件: {audio_file}")
        # 调用说话人分割器进行处理
        results = diarization(
            audio_file=audio_file,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            min_segment_duration=min_segment_duration,
            output_dir=output_dir,
            access_token=access_token,
            model_name_or_path=model_name_or_path,
            use_local_model=use_local_model
        )
        
        # 打印分割结果
        segments = results['segments']
        print(f"\n共识别到 {len(set(seg['speaker'] for seg in segments))} 个说话人, {len(segments)} 个片段:")
        
        for i, segment in enumerate(segments):
            print(f"片段 {i+1}: 说话人 {segment['speaker']}, "
                  f"时间: {segment['start']:.2f}s - {segment['end']:.2f}s, "
                  f"持续: {segment['duration']:.2f}s")
        
        # 如果输出目录已提供, 打印提取的音频路径
        if output_dir and 'speaker_audio_paths' in results:
            print("\n已提取音频片段:")
            for speaker, paths in results['speaker_audio_paths'].items():
                print(f"说话人 {speaker}: {len(paths)} 个片段")
                for path in paths[:3]:  # 只打印前3个路径
                    print(f"  - {path}")
                if len(paths) > 3:
                    print(f"  - ... 还有 {len(paths) - 3} 个片段")
        
        print("\n处理完成!")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()  # 打印详细错误信息


if __name__ == "__main__":
    main()
