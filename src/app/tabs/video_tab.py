#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr

def create_video_tab():
    """
    创建处理视频文件的标签页
    """
    with gr.Row():
        with gr.Column():
            video_input = gr.Video(
                label="输入视频"
            )
            
            with gr.Row():
                process_btn = gr.Button("处理视频")
                clear_btn = gr.Button("清除")
                
            options = gr.CheckboxGroup(
                choices=["检测人脸", "检测证件", "文本字幕"],
                label="处理选项",
                value=["检测人脸"]
            )
        
        with gr.Column():
            video_output = gr.Video(
                label="处理结果"
            )
            info = gr.JSON(label="处理信息")
    
    def process_video(video_path, options):
        if video_path is None:
            return None, {"状态": "失败", "原因": "无输入视频"}
        
        # 这里添加视频处理逻辑
        # processed_video = video_processor.process(video_path, options)
        
        return video_path, {  # 简单返回原视频作为示例
            "状态": "成功", 
            "选项": options,
            "视频路径": video_path
        }
    
    def clear():
        return None, None
    
    process_btn.click(
        fn=process_video,
        inputs=[video_input, options],
        outputs=[video_output, info]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=[],
        outputs=[video_input, video_output]
    )
    
    return [video_input, video_output, options, info]
