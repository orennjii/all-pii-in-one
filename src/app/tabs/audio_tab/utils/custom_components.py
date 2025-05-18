"""
自定义UI组件辅助工具

该模块提供用于创建和管理自定义UI组件的功能，
例如动态生成参考声音显示组件等。
"""

# System modules
import os
import glob
import gradio as gr


def show_custom_voices(reference_voices_dir, on_add_callback=None, on_delete_callback=None):
    """
    显示自定义声音列表
    
    参数:
        reference_voices_dir: 参考声音目录
        on_add_callback: 添加声音回调函数
        on_delete_callback: 删除声音回调函数
        
    返回:
        list: Gradio组件列表
    """
    # 获取自定义声音列表
    custom_voices = []
    if os.path.exists(reference_voices_dir):
        reference_voices = glob.glob(os.path.join(reference_voices_dir, "*.wav"))
        custom_voices = [path for path in reference_voices if os.path.basename(path).startswith("custom_")]
    
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
        if on_add_callback:
            add_btn.click(
                fn=on_add_callback,
                inputs=[gr.State(voice_path)],
                outputs=[]  # 输出将由回调函数定义
            )
        
        # 删除声音的回调
        if on_delete_callback:
            del_btn.click(
                fn=lambda path=voice_path: on_delete_callback(path),
                inputs=[],
                outputs=[]  # 输出将由回调函数定义
            )
        
        # 添加分隔线
        components.append(gr.Markdown("---"))
    
    return components


def create_voice_group(voice_paths, is_default=True, on_add_callback=None):
    """
    创建声音组显示
    
    参数:
        voice_paths: 声音文件路径列表
        is_default: 是否为默认声音组
        on_add_callback: 添加声音回调函数
        
    返回:
        list: Gradio组件列表
    """
    components = []
    
    if not voice_paths:
        return [gr.Markdown("没有可用的声音")]
    
    for voice_path in voice_paths:
        voice_name = os.path.basename(voice_path)
        if not is_default and voice_name.startswith("custom_"):
            voice_name = voice_name.replace("custom_", "")
            
        with gr.Accordion(voice_name, open=False) as accordion:
            components.append(accordion)
            components.append(gr.Audio(value=voice_path, label="预览", interactive=False))
            
            if on_add_callback:
                add_btn = gr.Button("添加到选择列表", elem_id=f"add_{voice_path}")
                components.append(add_btn)
                
                # 添加声音到选择列表的回调
                add_btn.click(
                    fn=lambda path=voice_path: on_add_callback(path),
                    inputs=[],
                    outputs=[]  # 输出将由回调函数定义
                )
    
    return components
