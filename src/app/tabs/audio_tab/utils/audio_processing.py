"""
音频处理工具函数

该模块提供音频处理相关的通用工具函数，包括参数解析、验证等。
"""

# System modules
import os


def parse_pitch_shifts(pitch_shifts_str):
    """
    解析音高调整参数字符串为整数列表
    
    参数:
        pitch_shifts_str: 音高调整参数字符串，如 "-2,1,0,3"
        
    返回:
        list: 整数列表，表示每个声音的音高调整值
        
    异常:
        ValueError: 如果参数格式不正确
    """
    if not pitch_shifts_str or pitch_shifts_str.strip() == "":
        return []
    
    try:
        return [int(x.strip()) for x in pitch_shifts_str.split(",")]
    except ValueError:
        raise ValueError("音高调整参数格式错误，请使用逗号分隔的整数列表，例如：-2,1,0")


def validate_audio_params(input_file, selected_voices, token):
    """
    验证音频处理参数
    
    参数:
        input_file: 输入音频文件路径
        selected_voices: 已选择的参考声音列表
        token: Hugging Face 认证令牌
        
    返回:
        tuple: (是否有效, 错误消息)
    """
    # 参数检查
    if not input_file or not os.path.exists(input_file):
        return False, "请先上传要处理的音频文件"
        
    if not token:
        return False, "请输入Hugging Face认证令牌，用于下载pyannote模型"
        
    # 如果选择列表为空，则提示用户选择参考声音
    if not selected_voices or len(selected_voices) == 0:
        return False, "请至少选择一个参考声音（可以通过参考声音下拉框选择，然后点击'添加到选择列表'按钮）"
    
    # 确保所有选择的参考声音文件都存在
    missing_files = []
    for voice_path in selected_voices:
        if not os.path.exists(voice_path):
            missing_files.append(os.path.basename(voice_path))
    
    if missing_files:
        return False, f"以下参考声音文件不存在: {', '.join(missing_files)}，请重新选择"
    
    return True, ""


def get_audio_duration(audio_file):
    """
    获取音频文件的时长
    
    参数:
        audio_file: 音频文件路径
        
    返回:
        float: 音频时长（秒）
    """
    try:
        import librosa
        duration = librosa.get_duration(path=audio_file)
        return duration
    except Exception as e:
        return 0.0
