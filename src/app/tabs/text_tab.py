#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr
import os
import sys

def create_text_tab():
    """
    创建处理文本文件的标签页
    """
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="输入文本",
                placeholder="请输入要处理的文本内容...",
                lines=10
            )
            text_file = gr.File(
                label="或上传文本文件",
                file_types=[".txt", ".md", ".json", ".csv"]
            )
            
            with gr.Row():
                process_btn = gr.Button("处理文本")
                clear_btn = gr.Button("清除")
        
        with gr.Column():
            text_output = gr.Textbox(
                label="处理结果",
                lines=10
            )
            info = gr.JSON(label="处理信息")
    
    def process_text(text, file_obj):
        if file_obj is not None:
            # 从文件读取文本
            content = file_obj.decode('utf-8')
        else:
            content = text
            
        if not content:
            return "请输入文本或上传文件", {"状态": "失败", "原因": "无输入内容"}
        
        # 这里添加文本处理逻辑
        # processed_text = text_processor.process(content)
        processed_text = f"已处理: {content[:100]}..." if len(content) > 100 else f"已处理: {content}"
        
        return processed_text, {"状态": "成功", "处理字符数": len(content)}
    
    def clear():
        return "", None, "", None
    
    process_btn.click(
        fn=process_text,
        inputs=[text_input, text_file],
        outputs=[text_output, info]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=[],
        outputs=[text_input, text_file, text_output, info]
    )
    
    return [text_input, text_file, text_output, info]
