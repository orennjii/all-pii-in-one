"""
参考声音文件管理工具

该模块提供用于管理参考声音文件的功能，包括上传、预览和删除参考声音。
"""

# System modules
import os
import shutil
import glob


def upload_reference_voice(audio, reference_voices_dir, voice_name=None):
    """
    上传参考声音
    
    参数:
        audio: 音频文件路径
        reference_voices_dir: 参考声音目录
        voice_name: 自定义声音名称（可选）
    
    返回:
        tuple: (文件路径, 状态消息, 更新后的选项)
    """
    if audio is None:
        return None, "未选择参考声音文件", []
    
    # 确保参考声音目录存在
    os.makedirs(reference_voices_dir, exist_ok=True)
    
    # 使用自定义名称或原始文件名
    if voice_name and voice_name.strip():
        # 确保文件名安全且有.wav后缀
        safe_name = "".join(c for c in voice_name if c.isalnum() or c in "._- ").strip()
        if not safe_name:
            safe_name = "custom_voice"
        if not safe_name.endswith(".wav"):
            safe_name += ".wav"
        basename = f"custom_{safe_name}"
    else:
        basename = f"custom_{os.path.basename(audio)}"
    
    # 保存上传的音频文件
    filename = os.path.join(reference_voices_dir, basename)
    shutil.copy(audio, filename)
    
    # 返回结果
    return filename, f"参考声音已上传：{basename}", []


def preview_reference_voice(voice_path):
    """
    返回参考声音的预览
    
    参数:
        voice_path: 参考声音文件路径
        
    返回:
        str: 音频文件路径，用于 Audio 组件预览
    """
    if not voice_path or not os.path.exists(voice_path):
        return None
    
    return voice_path


def delete_reference_voice(voice_path):
    """
    删除参考声音
    
    参数:
        voice_path: 参考声音文件路径
        
    返回:
        tuple: (状态消息, 更新后的选项)
    """
    # 只允许删除自定义上传的参考声音
    if not voice_path or not os.path.exists(voice_path) or not os.path.basename(voice_path).startswith("custom_"):
        return "只能删除自定义上传的参考声音", []
    
    try:
        os.remove(voice_path)
        return f"已删除参考声音: {os.path.basename(voice_path)}", []
    except Exception as e:
        return f"删除失败: {str(e)}", []


def get_reference_voices(reference_voices_dir, include_custom=True, include_default=True):
    """
    获取参考声音列表
    
    参数:
        reference_voices_dir: 参考声音目录路径
        include_custom: 是否包含自定义声音
        include_default: 是否包含默认声音
        
    返回:
        list: 参考声音文件路径列表
    """
    if not os.path.exists(reference_voices_dir):
        return []
        
    reference_voices = glob.glob(os.path.join(reference_voices_dir, "*.wav"))
    result = []
    
    if include_default:
        # 添加默认参考声音
        result.extend([path for path in reference_voices 
                      if not os.path.basename(path).startswith("custom_")])
    
    if include_custom:
        # 添加自定义参考声音
        result.extend([path for path in reference_voices 
                      if os.path.basename(path).startswith("custom_")])
    
    return result


def get_reference_voices_dropdown_options(reference_voices_dir):
    """
    获取参考声音下拉列表选项
    
    参数:
        reference_voices_dir: 参考声音目录路径
        
    返回:
        list: 元组列表，每个元组包含(显示名称, 文件路径)
    """
    default_voices = get_reference_voices(reference_voices_dir, include_custom=False)
    custom_voices = get_reference_voices(reference_voices_dir, include_default=False)
    
    # 创建选项列表
    options = []
    
    # 添加默认参考声音
    if default_voices:
        options.extend([(os.path.basename(path), path) for path in default_voices])
    
    # 添加自定义参考声音
    if custom_voices:
        options.extend([(os.path.basename(path).replace("custom_", ""), path) 
                       for path in custom_voices])
    
    return options
