#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gradio as gr

def create_image_tab():
    """
    创建处理图像文件的标签页
    """
    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="输入图像",
                type="pil"
            )
            
            with gr.Row():
                process_btn = gr.Button("处理图像")
                clear_btn = gr.Button("清除")
                
            options = gr.CheckboxGroup(
                choices=["检测文本", "检测人脸", "检测证件"],
                label="处理选项",
                value=["检测文本", "检测人脸"]
            )
        
        with gr.Column():
            image_output = gr.Image(
                label="处理结果"
            )
            info = gr.JSON(label="处理信息")
    
    def process_image(image, options):
        if image is None:
            return None, {"状态": "失败", "原因": "无输入图像"}
        
        # 这里添加图像处理逻辑
        # processed_image = image_processor.process(image, options)
        processed_image = image  # 简单返回原图作为示例
        
        return processed_image, {
            "状态": "成功", 
            "选项": options,
            "图像尺寸": f"{image.width}x{image.height}"
        }
    
    def clear():
        return None, None
    
    process_btn.click(
        fn=process_image,
        inputs=[image_input, options],
        outputs=[image_output, info]
    )
    
    clear_btn.click(
        fn=clear,
        inputs=[],
        outputs=[image_input, image_output]
    )
    
    return [image_input, image_output, options, info]
