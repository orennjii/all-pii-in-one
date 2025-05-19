"""
UI交互辅助工具

该模块提供用于UI组件交互的辅助功能，主要用于管理参考声音选择、
添加/移除声音、更新UI组件状态等。
"""

# System modules
import os

# External modules
import gradio as gr


def clear_selected_voices():
    """
    清空所选参考声音列表
    
    返回:
        tuple: (空列表, gr.update()更新) 用于重置状态和UI
    """
    return [], gr.update(choices=[])


def remove_selected_voices(selected_voices, dropdown_selected):
    """
    移除选中的参考声音
    
    参数:
        selected_voices: 当前已选择的声音列表
        dropdown_selected: 从下拉框中选择的要移除的声音
    
    返回:
        tuple: (更新后的声音列表, gr.update更新后的标签列表)
    """
    if not dropdown_selected:
        return selected_voices, gr.update()
    
    # 处理非列表输入
    if not isinstance(dropdown_selected, list):
        dropdown_selected = [dropdown_selected]
    
    # 移除选中的声音，确保只处理字符串类型
    updated_selected = []
    for v in selected_voices:
        if isinstance(v, str) and v not in dropdown_selected:
            updated_selected.append(v)
    
    # 更新标签
    labels = []
    for voice_path in updated_selected:
        if isinstance(voice_path, str):
            voice_name = os.path.basename(voice_path)
            if os.path.basename(voice_path).startswith("custom_"):
                voice_name = os.path.basename(voice_path).replace("custom_", "")
            labels.append((voice_name, voice_path))
    
    return updated_selected, gr.update(choices=labels)


def add_to_selection(
        selected: gr.State,
        dropdown_value,
        dropdown_options
    ):
    """
    添加参考声音到选择列表
    参数:
        selected (gr.State): 当前已选择的声音列表
        dropdown_value: 从下拉框中选择的要添加的声音
        dropdown_options: 下拉框中的所有选项
        
    返回:
        tuple: (更新后的声音列表, 带有gr.update的标签更新)
    """
    if not dropdown_value:
        return selected, gr.update(choices=[])
    
    # 检查值的类型并进行处理
    if isinstance(dropdown_value, list):
        # 如果是列表，递归处理每个元素
        updated_selected = selected.copy()
        for value in dropdown_value:
            updated_selected, _ = add_to_selection(updated_selected, value, dropdown_options)
        
        # 重新生成标签
        labels = []
        for voice_path in updated_selected:
            # 确保是字符串
            if isinstance(voice_path, str):
                voice_name = os.path.basename(voice_path)
                if voice_name.startswith("custom_"):
                    voice_name = voice_name.replace("custom_", "")
                labels.append((voice_name, voice_path))
        return updated_selected, gr.update(choices=labels)
    
    # 检查是否已经选择了这个声音
    if dropdown_value in selected:
        return selected, gr.update()
    
    # 添加到选择列表
    updated_selected = selected + [dropdown_value]
    
    # 获取所有选择项的标签
    labels = []
    for voice_path in updated_selected:
        # 确保是字符串
        if not isinstance(voice_path, str):
            continue
            
        # 查找对应的标签
        voice_name = os.path.basename(voice_path)
        if isinstance(dropdown_options, list) and isinstance(dropdown_options[0], dict):
            for group in dropdown_options:
                for label, path in group.get("options", []):
                    if path == voice_path:
                        voice_name = label
                        break
        elif isinstance(dropdown_options, list):
            for label, path in dropdown_options:
                if path == voice_path:
                    voice_name = label
                    break
        
        if os.path.basename(voice_path).startswith("custom_"):
            voice_name = os.path.basename(voice_path).replace("custom_", "")
            
        labels.append((voice_name, voice_path))
    
    return updated_selected, gr.update(choices=labels)


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
    
    # 处理列表输入
    if isinstance(current_selected, list):
        updated_selected = selected_voices.copy() if selected_voices else []
        for item in current_selected:
            updated_selected = update_selected_voices(updated_selected, item)
        return updated_selected
        
    # 非字符串输入检查
    if not isinstance(current_selected, str):
        return selected_voices
    
    # 如果声音已在列表中，则移除；否则添加
    if current_selected in selected_voices:
        return [v for v in selected_voices if v != current_selected]
    else:
        return selected_voices + [current_selected]


def add_custom_voice_to_selection(voice_path, selected_voices):
    """
    添加自定义声音到选择列表
    
    参数:
        voice_path: 参考声音文件路径
        selected_voices: 当前已选择的声音列表
        
    返回:
        tuple: (更新后的声音列表, gr.update更新标签)
    """
    # 处理列表输入
    if isinstance(voice_path, list):
        # 如果是列表，递归处理每个元素
        updated_selected = selected_voices.copy() if selected_voices else []
        for path in voice_path:
            updated_selected, _ = add_custom_voice_to_selection(path, updated_selected)
        
        # 重新生成标签
        labels = []
        for path in updated_selected:
            if isinstance(path, str):
                voice_name = os.path.basename(path)
                if voice_name.startswith("custom_"):
                    voice_name = voice_name.replace("custom_", "")
                labels.append((voice_name, path))
        return updated_selected, gr.update(choices=labels)
    
    # 单个路径情况
    if not voice_path or not isinstance(voice_path, str) or not os.path.exists(voice_path):
        return selected_voices, gr.update()
    
    # 如果已经在列表中，不重复添加
    if voice_path in selected_voices:
        # 返回现有列表和标签
        labels = []
        for path in selected_voices:
            if isinstance(path, str):
                voice_name = os.path.basename(path)
                if os.path.basename(path).startswith("custom_"):
                    voice_name = os.path.basename(path).replace("custom_", "") 
                labels.append((voice_name, path))
        return selected_voices, gr.update(choices=labels)
    
    # 添加到选择列表
    updated_selected = selected_voices + [voice_path]
    
    # 创建标签
    labels = []
    for path in updated_selected:
        if isinstance(path, str):
            voice_name = os.path.basename(path)
            if os.path.basename(path).startswith("custom_"):
                voice_name = os.path.basename(path).replace("custom_", "") 
            labels.append((voice_name, path))
        
    return updated_selected, gr.update(choices=labels)
