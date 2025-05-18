# System modules
import os
import glob
import tempfile
import shutil
import argparse

# External modules
import torch
import torchaudio
import numpy as np
import gradio as gr
from pyannote.audio import Pipeline
from pydub import AudioSegment
import librosa
import yaml
import soundfile as sf

# Internal modules
from src.modules.seed_vc_modules import SeedVCWrapper
from src.modules.seed_vc_modules.commons import str2bool
from src.commons.device_config import get_device
from src.processors.audio.audio_processor.voice_conversion import initialize_seed_vc

def get_audio_duration(file_path):
    """获取音频文件的时长（秒）"""
    audio_info = torchaudio.info(file_path)
    return audio_info.num_frames / audio_info.sample_rate

def create_temp_dir():
    """创建临时目录"""
    temp_dir = tempfile.mkdtemp(prefix="audio_anonymizer_")
    return temp_dir

def get_default_reference_voices():
    """获取默认参考声音列表"""
    reference_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples/reference")
    reference_voices = glob.glob(os.path.join(reference_dir, "*.wav"))
    # 过滤掉自定义上传的参考声音
    default_voices = [path for path in reference_voices if not os.path.basename(path).startswith("custom_")]
    return default_voices

def get_custom_reference_voices():
    """获取用户上传的自定义参考声音列表"""
    reference_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples/reference")
    reference_voices = glob.glob(os.path.join(reference_dir, "*.wav"))
    # 只返回自定义上传的参考声音
    custom_voices = [path for path in reference_voices if os.path.basename(path).startswith("custom_")]
    return custom_voices

def get_reference_voices_dropdown_options():
    """获取参考声音下拉列表选项"""
    default_voices = get_default_reference_voices()
    custom_voices = get_custom_reference_voices()
    
    # 创建选项列表，包括分组
    options = []
    
    # 添加默认参考声音组
    if default_voices:
        options.extend([(os.path.basename(path), path) for path in default_voices])
    
    # 添加自定义参考声音组
    if custom_voices:
        options.extend([(os.path.basename(path).replace("custom_", ""), path) for path in custom_voices])
    
    return options

# 使用src.commons.device_config中的get_device函数代替

def initialize_pipelines(auth_token, device):
    """
    初始化 pyannote 说话人分割管道
    
    参数:
        auth_token: Hugging Face 认证令牌
        device: 运行设备
    
    返回:
        Pipeline: 初始化后的 pyannote 管道
    """
    # 加载 pyannote 管道进行说话人分割
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=auth_token
    )
    
    pipeline.to(device)
    return pipeline

# 使用src.processors.audio.audio_processor.voice_conversion中的initialize_seed_vc函数代替

def perform_speaker_diarization(pipeline, input_file):
    """
    执行说话人分割
    
    参数:
        pipeline: 初始化后的 pyannote 管道
        input_file: 输入音频文件路径
    
    返回:
        object: 分割结果
        list: 说话人列表
    """
    # 应用预训练的管道进行说话人分割
    diarization = pipeline(input_file)
    
    # 获取所有唯一的说话人标签
    speakers = set()
    for _, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    speakers = sorted(list(speakers))
    
    return diarization, speakers

def create_speaker_mapping(speakers, reference_voices, pitch_shifts):
    """
    创建说话人到参考声音和音高调整的映射
    
    参数:
        speakers: 说话人列表
        reference_voices: 参考声音文件路径列表
        pitch_shifts: 音高调整列表
    
    返回:
        dict: 说话人到参考声音的映射
        dict: 说话人到音高调整的映射
        str: 说话人信息文本
    """
    # 确保有足够的参考声音
    if len(reference_voices) < len(speakers):
        raise ValueError(f"参考声音数量 ({len(reference_voices)}) 少于检测到的说话人数量 ({len(speakers)})")
    
    # 创建说话人到参考声音的映射
    speaker_to_voice = {speaker: reference_voices[i] for i, speaker in enumerate(speakers)}
    speaker_to_pitch = {speaker: pitch_shifts[i] for i, speaker in enumerate(speakers)}
    
    speaker_info = "说话人映射关系:\n"
    for speaker, voice in speaker_to_voice.items():
        speaker_info += f"  {speaker} -> {os.path.basename(voice)} (音高调整: {speaker_to_pitch[speaker]}半音)\n"
    
    return speaker_to_voice, speaker_to_pitch, speaker_info

def extract_and_convert_segments(
    diarization, waveform, sample_rate, speaker_to_voice, speaker_to_pitch,
    segments_dir, vc_model, diffusion_steps, length_adjust, inference_cfg_rate,
    f0_condition, use_auto_f0_adjust, input_file
):
    """
    提取并转换每个说话人片段
    
    参数:
        diarization: 分割结果
        waveform: 音频波形
        sample_rate: 采样率
        speaker_to_voice: 说话人到参考声音的映射
        speaker_to_pitch: 说话人到音高调整的映射
        segments_dir: 片段保存目录
        vc_model: Seed-VC 模型
        diffusion_steps: 扩散步数
        length_adjust: 长度调整因子
        inference_cfg_rate: 推理 CFG 速率
        f0_condition: 是否使用 F0 条件
        use_auto_f0_adjust: 是否自动调整 F0
        input_file: 输入音频文件路径
    
    返回:
        list: 片段信息列表
        int: 片段总数
    """
    segments = []
    
    # 获取分割片段数量
    segments_count = 0
    for _ in diarization.itertracks(yield_label=True):
        segments_count += 1
    
    # 处理各个说话人片段
    for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
        # 计算帧数
        start_frame = int(turn.start * sample_rate)
        end_frame = int(turn.end * sample_rate)
        
        # 提取片段
        segment = waveform[:, start_frame:end_frame]
        
        # 保存原始片段
        segment_path = os.path.join(segments_dir, f"segment_{i}_speaker_{speaker}.wav")
        torchaudio.save(segment_path, segment, sample_rate)
        
        # 转换声音并保存
        converted_path = os.path.join(segments_dir, f"converted_{i}_speaker_{speaker}.wav")
        segments.append(convert_segment(
            segment_path, converted_path, speaker, 
            speaker_to_voice, speaker_to_pitch,
            vc_model, diffusion_steps, length_adjust, 
            inference_cfg_rate, f0_condition, use_auto_f0_adjust,
            turn.start, turn.end
        ))
    
    return segments, segments_count

def convert_segment(
    segment_path, converted_path, speaker, 
    speaker_to_voice, speaker_to_pitch,
    vc_model, diffusion_steps, length_adjust, 
    inference_cfg_rate, f0_condition, use_auto_f0_adjust,
    start_time, end_time
):
    """
    转换单个音频片段
    
    参数:
        segment_path: 原始片段路径
        converted_path: 转换后片段保存路径
        speaker: 说话人标签
        speaker_to_voice: 说话人到参考声音的映射
        speaker_to_pitch: 说话人到音高调整的映射
        vc_model: Seed-VC 模型
        diffusion_steps: 扩散步数
        length_adjust: 长度调整因子
        inference_cfg_rate: 推理 CFG 速率
        f0_condition: 是否使用 F0 条件
        use_auto_f0_adjust: 是否自动调整 F0
        start_time: 片段开始时间
        end_time: 片段结束时间
    
    返回:
        dict: 片段信息
    """
    # 选择对应的参考声音和音高调整
    reference_voice = speaker_to_voice[speaker]
    pitch_shift = speaker_to_pitch[speaker]
    
    # 使用 SeedVC 进行声音转换
    converted_audio = vc_model.convert_voice(
        segment_path,
        reference_voice,
        diffusion_steps=diffusion_steps,
        length_adjust=length_adjust,
        inference_cfg_rate=inference_cfg_rate,
        f0_condition=f0_condition,
        auto_f0_adjust=use_auto_f0_adjust,
        pitch_shift=pitch_shift,
        stream_output=False
    )
    
    # 保存为 WAV 文件
    sr = 44100 if f0_condition else 22050
    sf.write(converted_path, converted_audio, sr)
    
    # 返回片段信息
    return {
        'start': start_time,
        'end': end_time,
        'file': converted_path
    }

def merge_segments(segments, output_file):
    """
    合并所有转换后的片段
    
    参数:
        segments: 片段信息列表
        output_file: 输出文件路径
    
    返回:
        str: 输出文件路径
    """
    # 读取第一个片段作为基础
    combined = AudioSegment.from_wav(segments[0]['file'])
    
    # 添加其他片段
    for i in range(1, len(segments)):
        # 计算当前片段与上一个片段之间的静音时长（毫秒）
        silence_duration = (segments[i]['start'] - segments[i-1]['end']) * 1000
        
        # 如果有间隙，添加静音
        if silence_duration > 0:
            silence = AudioSegment.silent(duration=silence_duration)
            combined += silence
            
        # 添加当前片段
        current_segment = AudioSegment.from_wav(segments[i]['file'])
        combined += current_segment
    
    # 导出为最终音频文件
    combined.export(output_file, format="wav")
    
    return output_file

def anonymize_audio(
    input_file, 
    reference_voices, 
    auth_token, 
    diffusion_steps=10, 
    length_adjust=1.0, 
    inference_cfg_rate=0.7, 
    f0_condition=True,
    use_auto_f0_adjust=True,
    pitch_shifts=None,
    progress=gr.Progress()
):
    """
    对音频文件进行匿名化处理，将不同说话人的声音替换为参考声音
    
    参数:
        input_file: 输入音频文件路径
        reference_voices: 参考声音文件路径列表
        auth_token: Hugging Face 认证令牌
        diffusion_steps: 扩散步数（默认：10）
        length_adjust: 长度调整因子（默认：1.0）
        inference_cfg_rate: 推理 CFG 速率（默认：0.7）
        f0_condition: 是否使用 F0 条件（默认：True）
        use_auto_f0_adjust: 是否自动调整 F0（默认：True）
        pitch_shifts: 各个声音的音高调整，单位为半音（默认：None，表示不调整）
        progress: gradio 进度条
    """
    # 创建临时目录存放分割的音频片段和输出文件
    temp_dir = create_temp_dir()
    segments_dir = os.path.join(temp_dir, "segments")
    os.makedirs(segments_dir, exist_ok=True)
    
    output_file = os.path.join(temp_dir, "anonymized_output.wav")
    
    # 根据需要设置音高调整
    if pitch_shifts is None or len(pitch_shifts) == 0:
        pitch_shifts = [0] * len(reference_voices)
    elif len(pitch_shifts) < len(reference_voices):
        pitch_shifts.extend([0] * (len(reference_voices) - len(pitch_shifts)))
    
    # 获取设备
    device = get_device()
    
    # 初始化模型
    progress(0, desc="正在加载 pyannote 模型...")
    pipeline = initialize_pipelines(auth_token, device)

    # 进行说话人分割
    progress(0.1, desc="正在对音频进行说话人分割...")
    diarization, speakers = perform_speaker_diarization(pipeline, input_file)
    
    progress(0.2, desc="正在加载 Seed-VC 模型...")
    vc_model = initialize_seed_vc(device)
    
    # 加载音频文件
    waveform, sample_rate = torchaudio.load(input_file)
    
    progress(0.3, desc=f"检测到 {len(speakers)} 个说话人")
    
    # 创建说话人映射
    speaker_to_voice, speaker_to_pitch, speaker_info = create_speaker_mapping(
        speakers, reference_voices, pitch_shifts
    )
    
    # 提取并转换片段
    segments, segments_count = extract_and_convert_segments(
        diarization, waveform, sample_rate, speaker_to_voice, speaker_to_pitch,
        segments_dir, vc_model, diffusion_steps, length_adjust, inference_cfg_rate,
        f0_condition, use_auto_f0_adjust, input_file
    )
    
    # 处理进度条更新
    total_duration = get_audio_duration(input_file)
    processed_duration = 0
    
    # 重新处理每个片段的进度更新
    for i, segment in enumerate(segments):
        segment_duration = segment['end'] - segment['start']
        processed_duration += segment_duration
        segment_progress = processed_duration / total_duration
        
        # 更新进度条，从 0.3 到 0.8 之间
        progress_value = 0.3 + (segment_progress * 0.5)
        progress(progress_value, desc=f"处理片段 {i+1}/{segments_count}: {segment['start']:.2f}s 到 {segment['end']:.2f}s")
    
    # 合并片段
    progress(0.8, desc="正在合并所有转换后的片段...")
    merge_segments(segments, output_file)
    
    progress(0.9, desc="正在导出最终音频文件...")
    progress(1.0, desc="处理完成！")
    
    return output_file, speaker_info

def parse_pitch_shifts(pitch_shifts_str):
    """解析音高调整参数"""
    if not pitch_shifts_str or pitch_shifts_str.strip() == "":
        return []
    
    try:
        return [int(x.strip()) for x in pitch_shifts_str.split(",")]
    except ValueError:
        raise ValueError("音高调整参数格式错误，请使用逗号分隔的整数列表，例如：-2,1,0")

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
        return None, "未选择参考声音文件", [], get_reference_voices_dropdown_options()
    
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
    
    # 获取更新后的参考声音列表
    return filename, f"参考声音已上传：{basename}", [], get_reference_voices_dropdown_options()

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
        return "只能删除自定义上传的参考声音", get_reference_voices_dropdown_options()
    
    try:
        os.remove(voice_path)
        return f"已删除参考声音: {os.path.basename(voice_path)}", get_reference_voices_dropdown_options()
    except Exception as e:
        return f"删除失败: {str(e)}", get_reference_voices_dropdown_options()

def update_selected_voices(selected_voices, current_selected):
    """
    更新已选择的参考声音列表
    
    参数:
        selected_voices: 当前已选择的参考声音列表
        current_selected: 当前选择的声音路径
        
    返回:
        list: 更新后的选择列表
    """
    if not current_selected:
        return selected_voices
    
    # 如果声音已在列表中，则移除；否则添加
    if current_selected in selected_voices:
        return [v for v in selected_voices if v != current_selected]
    else:
        return selected_voices + [current_selected]

def initialize_gradio_ui(reference_voices_dir):
    """
    创建 Gradio 用户界面
    
    参数:
        reference_voices_dir: 参考声音目录路径
    
    返回:
        gr.Blocks: Gradio 界面
    """
    with gr.Blocks(title="音频匿名化工具") as demo:
        # 存储选择的参考声音列表
        selected_voices_state = gr.State([])
        
        gr.Markdown("""
        # 音频匿名化工具
        
        此工具结合了 pyannote 和 Seed-VC，可以对音频中的多个说话人进行声音替换，实现音频脱敏。
        
        ## 使用流程:
        1. 上传需要处理的音频文件
        2. 选择或上传参考声音文件 (至少与音频中说话人数量相同)
        3. 设置处理参数
        4. 点击"开始处理"按钮
        """)
        
        with gr.Tab("基本设置"):
            with gr.Row():
                with gr.Column():
                    input_audio = gr.Audio(type="filepath", label="输入音频文件")
                    auth_token = gr.Textbox(
                        label="Hugging Face 认证令牌", 
                        placeholder="hf_...", 
                        info="用于下载 pyannote 模型，需要在 Hugging Face 上申请"
                    )
                
                # 参考声音选择
                with gr.Column():
                    # 获取参考声音下拉选项
                    dropdown_options = get_reference_voices_dropdown_options()
                    
                    # 创建参考声音下拉列表
                    reference_dropdown = gr.Dropdown(
                        choices=dropdown_options,
                        label="添加参考声音",
                        info="从下拉列表中选择要添加的参考声音",
                        type="value",
                        multiselect=True,
                    )
                    
                    # 已选择的参考声音列表
                    reference_voices = gr.Dropdown(
                        choices=[],
                        label="已选择的参考声音",
                        info="已选择的参考声音列表",
                        multiselect=True,
                        interactive=True,
                    )
                    
                    # 添加按钮、移除按钮和清空按钮
                    with gr.Row():
                        add_voice_button = gr.Button("添加到选择列表", variant="primary")
                        remove_voice_button = gr.Button("移除选中的参考声音", variant="stop")
                        clear_voice_button = gr.Button("清空选择", variant="secondary")
                    
                    # 清空选择的回调
                    def clear_selected_voices():
                        return [], []
                    
                    clear_voice_button.click(
                        fn=clear_selected_voices,
                        inputs=[],
                        outputs=[selected_voices_state, reference_voices]
                    )
                    
                    # 移除选中的参考声音的回调
                    def remove_selected_voices(selected_voices, dropdown_selected):
                        if not dropdown_selected:
                            return selected_voices, []
                        
                        # 移除选中的声音
                        updated_selected = [v for v in selected_voices if v not in dropdown_selected]
                        
                        # 更新标签
                        labels = []
                        for voice_path in updated_selected:
                            voice_name = os.path.basename(voice_path)
                            if os.path.basename(voice_path).startswith("custom_"):
                                voice_name = os.path.basename(voice_path).replace("custom_", "")
                            labels.append((voice_name, voice_path))
                        
                        return updated_selected, labels
                    
                    remove_voice_button.click(
                        fn=remove_selected_voices,
                        inputs=[selected_voices_state, reference_voices],
                        outputs=[selected_voices_state, reference_voices]
                    )

                    # 添加参考声音到选择列表的回调
                    def add_to_selection(selected, dropdown_value, dropdown_options):
                        if not dropdown_value:
                            return selected, []
                        
                        # 检查是否已经选择了这个声音
                        if dropdown_value in selected:
                            return selected, []
                        
                        # 添加到选择列表
                        updated_selected = selected + [dropdown_value]
                        
                        # 获取所有选择项的标签
                        labels = []
                        for voice_path in updated_selected:
                            # 查找对应的标签
                            voice_name = os.path.basename(voice_path)
                            for group in dropdown_options:
                                for label, path in group["options"]:
                                    if path == voice_path:
                                        voice_name = label
                                        break
                            labels.append((voice_name, voice_path))
                        
                        return updated_selected, labels
                    
                    add_voice_button.click(
                        fn=add_to_selection,
                        inputs=[selected_voices_state, reference_dropdown, gr.State(dropdown_options)],
                        outputs=[selected_voices_state, reference_voices]
                    )
                    
                    # 上传自定义参考声音
                    gr.Markdown("### 上传自定义参考声音")
                    with gr.Row():
                        with gr.Column(scale=2):
                            custom_reference = gr.Audio(type="filepath", label="选择音频文件")
                        with gr.Column(scale=1):
                            custom_reference_name = gr.Textbox(
                                label="自定义声音名称（可选）", 
                                placeholder="给声音起一个名字，如果不填则使用文件名"
                            )
                            upload_button = gr.Button("上传参考声音", variant="primary")
                    
                    upload_status = gr.Textbox(label="上传状态", interactive=False)
                    
                    # 上传参考声音的回调
                    def upload_and_refresh(audio, name):
                        result = upload_reference_voice(audio, reference_voices_dir, name)
                        # 刷新下拉列表和自定义声音容器（如果已加载）
                        updated_dropdown = get_reference_voices_dropdown_options()
                        # 尝试更新自定义声音容器，但不强制要求它存在
                        try:
                            custom_components = show_custom_voices()
                            return result[0], result[1], "", updated_dropdown, custom_components
                        except:
                            # 如果自定义声音容器尚未加载，则只返回前面的值
                            return result[0], result[1], "", updated_dropdown
                    
                    # 定义一个包装函数来处理不同的输出情况
                    def upload_wrapper(audio, name):
                        results = upload_and_refresh(audio, name)
                        # 只返回前四个值，忽略可能的第五个值
                        return results[0], results[1], results[2], results[3]
                    
                    upload_button.click(
                        fn=upload_wrapper,
                        inputs=[custom_reference, custom_reference_name],
                        outputs=[custom_reference, upload_status, custom_reference_name, reference_dropdown]
                    )
            
            with gr.Row():
                pitch_shifts = gr.Textbox(
                    label="音高调整（半音）", 
                    placeholder="例如：-2,1,0,3", 
                    info="用逗号分隔的整数列表，与参考声音一一对应。正值升高音调，负值降低音调。"
                )
            
            with gr.Row():
                diffusion_steps = gr.Slider(
                    minimum=1, maximum=100, value=10, step=1, 
                    label="扩散步数", 
                    info="影响声音转换质量，值越大质量越好但处理越慢"
                )
                length_adjust = gr.Slider(
                    minimum=0.5, maximum=2.0, step=0.1, value=1.0, 
                    label="长度调整因子", 
                    info="<1.0 加速语速，>1.0 减慢语速"
                )
            
            with gr.Row():
                inference_cfg_rate = gr.Slider(
                    minimum=0.0, maximum=1.0, step=0.1, value=0.7, 
                    label="推理 CFG 速率", 
                    info="对声音质量有微小影响"
                )
                f0_condition = gr.Checkbox(
                    label="使用 F0 条件", 
                    value=True, 
                    info="保留音高信息，通常应该启用"
                )
                auto_f0_adjust = gr.Checkbox(
                    label="自动调整 F0", 
                    value=True, 
                    info="自动调整音高以匹配参考声音"
                )
        
        with gr.Tab("参考声音库"):
            gr.Markdown("### 参考声音库\n\n在此标签页中，您可以预览和管理所有可用的参考声音。")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### 默认参考声音")
                    # 获取默认参考声音列表
                    default_voices = get_default_reference_voices()
                    
                    # 为每个默认参考声音创建一个可折叠组件
                    for voice_path in default_voices:
                        voice_name = os.path.basename(voice_path)
                        with gr.Accordion(voice_name, open=False):
                            gr.Audio(value=voice_path, label="预览", interactive=False)
                            with gr.Row():
                                add_default_btn = gr.Button("添加到选择列表", elem_id=f"add_{voice_path}")
                                
                                # 添加默认声音到选择列表的回调
                                add_default_btn.click(
                                    fn=add_to_selection,
                                    inputs=[selected_voices_state, gr.State(voice_path), gr.State(dropdown_options)],
                                    outputs=[selected_voices_state, reference_voices]
                                )
                
                with gr.Column(scale=1):
                    gr.Markdown("#### 自定义参考声音")
                    
                    # 上传新的参考声音
                    with gr.Row(equal_height=True):
                        new_custom_reference = gr.Audio(type="filepath", label="上传新的参考声音")
                        new_custom_name = gr.Textbox(
                            label="声音名称", 
                            placeholder="给声音起一个名字，如果不填则使用文件名"
                        )
                    
                    new_upload_button = gr.Button("上传参考声音")
                    new_upload_status = gr.Textbox(label="上传状态", interactive=False)
                    
                    # 分隔线
                    gr.Markdown("---")
                    
                    # 创建自定义声音动态组件（先定义，后使用）
                    custom_voices_container = gr.Column(elem_id="custom_voices_container")
                    
                    # 上传新参考声音的回调，全局更新所有相关组件
                    def upload_and_refresh_all(audio, name):
                        if not audio:
                            return None, "请先选择一个音频文件", "", None, None
                        
                        result = upload_reference_voice(audio, reference_voices_dir, name)
                        # 刷新下拉列表和自定义声音显示
                        updated_dropdown = get_reference_voices_dropdown_options()
                        custom_components = show_custom_voices()
                        return result[0], result[1], "", updated_dropdown, custom_components
                    
                    new_upload_button.click(
                        fn=upload_and_refresh_all,
                        inputs=[new_custom_reference, new_custom_name],
                        outputs=[new_custom_reference, new_upload_status, new_custom_name, reference_dropdown, custom_voices_container]
                    )
                    
                    # 显示自定义声音列表的函数
                    def show_custom_voices():
                        custom_voices = get_custom_reference_voices()
                        components = []
                        
                        if not custom_voices:
                            return [gr.Markdown("暂无自定义参考声音")]
                        
                        for voice_path in custom_voices:
                            voice_name = os.path.basename(voice_path).replace("custom_", "")
                            components.append(gr.Markdown(f"### {voice_name}"))
                            components.append(gr.Audio(value=voice_path, label="预览"))
                            
                            row = gr.Row()
                            with row:
                                add_btn = gr.Button("添加到选择列表")
                                del_btn = gr.Button("删除此声音")
                            components.append(row)
                            
                            # 添加到选择列表的回调
                            add_btn.click(
                                fn=add_custom_voice_to_selection,
                                inputs=[gr.State(voice_path), selected_voices_state],
                                outputs=[selected_voices_state, reference_voices]
                            )
                            
                            # 删除声音的回调
                            def delete_and_refresh(voice_path, selected_voices):
                                message = delete_reference_voice(voice_path)
                                new_dropdown_options = get_reference_voices_dropdown_options()
                                
                                # 如果被删除的声音在已选择列表中，则移除它
                                if voice_path in selected_voices:
                                    selected_voices = [v for v in selected_voices if v != voice_path]
                                
                                # 更新已选择的声音列表显示
                                labels = []
                                for vp in selected_voices:
                                    voice_name = os.path.basename(vp)
                                    if os.path.basename(vp).startswith("custom_"):
                                        voice_name = os.path.basename(vp).replace("custom_", "")
                                    labels.append((voice_name, vp))
                                
                                return message[0], show_custom_voices(), new_dropdown_options, selected_voices, labels
                            
                            del_btn.click(
                                fn=lambda sv=selected_voices_state: delete_and_refresh(voice_path, sv),
                                inputs=[],
                                outputs=[new_upload_status, custom_voices_container, reference_dropdown, 
                                         selected_voices_state, reference_voices]
                            )
                            
                            # 添加分隔线
                            components.append(gr.Markdown("---"))
                        
                        return components
                    
                    # 初始化展示自定义声音
                    with custom_voices_container:
                        custom_components = show_custom_voices()
                    
                    # 刷新自定义声音列表按钮
                    refresh_custom_btn = gr.Button("刷新自定义声音列表")
                    
                    # 刷新自定义声音列表回调
                    refresh_custom_btn.click(
                        fn=show_custom_voices,
                        inputs=[],
                        outputs=[custom_voices_container]
                    )
            
        with gr.Tab("高级选项"):
            gr.Markdown("""
            ## 高级使用技巧
            
            1. **分割质量调整**：如果说话人分割不准确，可以尝试使用更专业的音频编辑软件进行预处理。
            
            2. **音质优化**：
               - 增加扩散步数可以提高音质，但会增加处理时间
               - 调整 inference_cfg_rate 可以微调输出音质
            
            3. **音高调整**：
               - 使用"音高调整"参数可以为每个参考声音单独设置音高偏移
               - 这对于让转换后的声音更自然非常有帮助
            
            4. **长度调整**：
               - 调整长度因子可以改变语速而不改变音调
               - 特别适合匹配不同语速的说话人
            """)
        
        with gr.Row():
            start_button = gr.Button("开始处理", variant="primary")
            
        with gr.Row():
            with gr.Column():
                output_audio = gr.Audio(label="输出音频")
            with gr.Column():
                speaker_info = gr.Textbox(label="说话人信息", interactive=False)
        
        # 添加示例
        example_reference_voices = [
            os.path.join(reference_voices_dir, "teio_0.wav"),
            os.path.join(reference_voices_dir, "trump_0.wav")
        ]
        
        gr.Examples(
            examples=[
                ["examples/reference/dingzhen_0.wav", "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",
                 example_reference_voices, "-2,1", 10, 1.0, 0.7, True, True],
            ],
            inputs=[input_audio, auth_token, reference_voices, pitch_shifts, 
                    diffusion_steps, length_adjust, inference_cfg_rate, 
                    f0_condition, auto_f0_adjust]
        )                    # 设置处理函数，带有更好的错误处理
        def process_audio(input_file, selected_voices, token, steps, length, cfg, f0, auto_f0, pitch):
            # 参数检查
            if not input_file or not os.path.exists(input_file):
                return None, "请先上传要处理的音频文件"
                
            if not token:
                return None, "请输入Hugging Face认证令牌，用于下载pyannote模型"
                
            # 如果选择列表为空，则提示用户选择参考声音
            if not selected_voices or len(selected_voices) == 0:
                return None, "请至少选择一个参考声音（可以通过参考声音下拉框选择，然后点击'添加到选择列表'按钮）"
            
            # 确保所有选择的参考声音文件都存在
            missing_files = []
            for voice_path in selected_voices:
                if not os.path.exists(voice_path):
                    missing_files.append(os.path.basename(voice_path))
            
            if missing_files:
                return None, f"以下参考声音文件不存在: {', '.join(missing_files)}，请重新选择"
            
            try:
                # 解析音高调整值，如果有的话
                parsed_pitch = parse_pitch_shifts(pitch) if pitch else []
                
                # 调用匿名化处理函数，确保它能捕获所有异常
                try:
                    return anonymize_audio(
                        input_file, 
                        selected_voices, 
                        token, 
                        diffusion_steps=steps, 
                        length_adjust=length, 
                        inference_cfg_rate=cfg, 
                        f0_condition=f0,
                        use_auto_f0_adjust=auto_f0,
                        pitch_shifts=parsed_pitch
                    )
                except ValueError as e:
                    return None, f"处理错误: {str(e)}"
                except Exception as e:
                    return None, f"发生未预期的错误: {str(e)}，请检查参数或联系开发者"
            except Exception as e:
                return None, f"参数错误: {str(e)}"
        
        start_button.click(
            fn=process_audio,
            inputs=[
                input_audio, 
                selected_voices_state,  # 使用保存的选择列表状态
                auth_token,
                diffusion_steps,
                length_adjust,
                inference_cfg_rate,
                f0_condition,
                auto_f0_adjust,
                pitch_shifts
            ],
            outputs=[output_audio, speaker_info]
        )
        
        # 设置处理函数
        # 处理函数
        def process_audio(input_file, selected_voices, token, steps, length, cfg, f0, auto_f0, pitch):
            # 如果选择列表为空，则提示用户选择参考声音
            if not selected_voices or len(selected_voices) == 0:
                return None, "请至少选择一个参考声音"
            
            # 选择的参考声音是值列表，每个值是路径
            voice_paths = selected_voices
            
            # 调用匿名化处理函数
            return anonymize_audio(
                input_file, 
                voice_paths, 
                token, 
                diffusion_steps=steps, 
                length_adjust=length, 
                inference_cfg_rate=cfg, 
                f0_condition=f0,
                use_auto_f0_adjust=auto_f0,
                pitch_shifts=parse_pitch_shifts(pitch)
            )
        
        start_button.click(
            fn=process_audio,
            inputs=[
                input_audio, 
                selected_voices_state,  # 使用保存的选择列表状态
                auth_token,
                diffusion_steps,
                length_adjust,
                inference_cfg_rate,
                f0_condition,
                auto_f0_adjust,
                pitch_shifts
            ],
            outputs=[output_audio, speaker_info]
        )
        
        # 添加自定义声音到选择列表
        def add_custom_voice_to_selection(voice_path, selected_voices):
            if not voice_path or not os.path.exists(voice_path):
                return selected_voices, []
            
            # 如果已经在列表中，不重复添加
            if voice_path in selected_voices:
                # 返回现有列表和标签
                labels = []
                for path in selected_voices:
                    voice_name = os.path.basename(path)
                    if os.path.basename(path).startswith("custom_"):
                        voice_name = os.path.basename(path).replace("custom_", "") 
                    labels.append((voice_name, path))
                return selected_voices, labels
            
            # 添加到选择列表
            updated_selected = selected_voices + [voice_path]
            
            # 创建标签
            labels = []
            for path in updated_selected:
                voice_name = os.path.basename(path)
                if os.path.basename(path).startswith("custom_"):
                    voice_name = os.path.basename(path).replace("custom_", "") 
                labels.append((voice_name, path))
                
            return updated_selected, labels
        
        # 自定义CSS样式
        custom_css = """
        .gradio-container .custom-voice-item {
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            background-color: #f9f9f9;
        }
        """
        
        demo.css = custom_css
    
    return demo

def main():
    """
    主函数，解析命令行参数并启动应用
    """
    # 设置参考声音目录
    reference_voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples/reference")
    
    # 定义命令行参数
    parser = argparse.ArgumentParser(description="音频匿名化工具 - Web 界面")
    parser.add_argument("--share", type=str2bool, nargs="?", const=True, default=False, help="是否共享应用")
    parser.add_argument("--gpu", type=int, help="使用的 GPU ID", default=0)
    args = parser.parse_args()
    
    # 设置设备
    cuda_target = f"cuda:{args.gpu}" if torch.cuda.is_available() and args.gpu is not None else "cuda"
    device = get_device()
    print(f"使用设备: {device}")
    
    # 创建并启动 Gradio 界面
    demo = initialize_gradio_ui(reference_voices_dir)
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
