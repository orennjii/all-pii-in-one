#!/usr/bin/env python
# -*- coding: utf-8 -*-

# System modules
import gradio as gr
import os
import sys

# Internal modules
from src.app.tabs.audio_tab.audio_ui import AudioUI
from src.processors.audio.audio_processor import AudioAnonymizer

# 导入处理模块 (后续可以根据需要添加)
# from src.processors.text import text_processor
# from src.processors.audio import audio_processor, speaker_diarizer, voice_converter
# from src.core.pattern_recognizer import pattern_recognizer

def create_audio_tab():
    """
    创建音频处理标签页的组件
    
    返回:
        dict: 包含音频处理组件的字典
    """
    # 设置参考声音目录
    reference_voices_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tabs/examples/reference")
    
    # 初始化音频匿名化处理器
    audio_anonymizer = AudioAnonymizer()
    
    # 初始化用户界面
    audio_ui = AudioUI(audio_anonymizer)
    audio_ui.set_reference_voices_dir(reference_voices_dir)
    
    # 创建并返回组件
    return audio_ui.create_tab_ui()

def initialize_app():
    """
    创建主应用
    """
    with gr.Blocks(title="ALL PII IN ONE", theme=gr.themes.Soft()) as app:
        gr.Markdown("# 综合PII编辑系统\n#### 支持多种文件类型的个人身份信息编辑")
        
        with gr.Tabs() as tabs_container:
            with gr.TabItem("文本处理"):
                gr.Markdown("文本处理功能正在开发中...")
                
            with gr.TabItem("图像处理"):
                gr.Markdown("图像处理功能正在开发中...")
                
            with gr.TabItem("音频处理"):
                audio_components = create_audio_tab()
                
            with gr.TabItem("视频处理"):
                gr.Markdown("视频处理功能正在开发中...")
        
        gr.Markdown("### 处理说明\n - 文本处理：支持纯文本中的PII检测与匿名化\n - 图像处理：支持图像中的文本和非文本PII编辑\n - 音频处理：支持说话人分割与语音匿名化\n - 视频处理：支持视频中的PII对象检测与编辑")
    
    return app

if __name__ == "__main__":
    app = initialize_app()
    app.launch(share=False)
