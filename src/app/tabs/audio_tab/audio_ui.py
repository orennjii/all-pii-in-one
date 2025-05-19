"""
音频匿名化用户界面 (简化版)

此版本演示了如何使用 utils 模块中的工具函数
来减少对 AudioUI 类实例状态的依赖。
"""

# System modules
import os
import glob

# External modules
import gradio as gr
import torch
from src.processors.audio_processor import AudioAnonymizer

# Local modules
from src.app.tabs.audio_tab.utils.voice_file_utils import (
    upload_reference_voice, 
    preview_reference_voice, 
    delete_reference_voice,
    get_reference_voices,
    get_reference_voices_dropdown_options
)
from src.app.tabs.audio_tab.utils.ui_helpers import (
    update_selected_voices, 
    add_custom_voice_to_selection,
    clear_selected_voices,
    remove_selected_voices,
    add_to_selection
)
from src.app.tabs.audio_tab.utils.audio_processing import (
    parse_pitch_shifts,
    validate_audio_params
)
from src.app.tabs.audio_tab.utils.custom_components import (
    show_custom_voices,
    create_voice_group
)


class AudioUI:
    """
    音频匿名化的用户界面类
    """

    def __init__(
        self,
        device: torch.device = None,
        reference_voices_dir: str = None,
    ):
        """
        初始化音频UI
        
        Args:
            device (torch.device): 设备类型(CPU或GPU)
            reference_voices_dir (str): 参考声音目录
        """
        self.audio_anonymizer = AudioAnonymizer(device=device)
        self.reference_voices_dir = reference_voices_dir
    
    def upload_reference_voice(self, audio, reference_voices_dir, voice_name=None):
        """
        上传参考声音
        
        参数:
            audio: 音频文件路径
            reference_voices_dir: 参考声音目录
            voice_name: 自定义声音名称（可选）
        
        返回:
            tuple: (文件路径, 状态消息, 更新后的选项)
        """
        result = upload_reference_voice(audio, reference_voices_dir, voice_name)
        return result[0], result[1], [], get_reference_voices_dropdown_options(reference_voices_dir)
    
    def preview_reference_voice(self, voice_path):
        """返回参考声音的预览"""
        return preview_reference_voice(voice_path)
    
    def delete_reference_voice(self, voice_path):
        """删除参考声音"""
        result = delete_reference_voice(voice_path)
        return result[0], get_reference_voices_dropdown_options(self.reference_voices_dir)

    def update_selected_voices(self, selected_voices, current_selected):
        """更新已选择的参考声音列表"""
        return update_selected_voices(selected_voices, current_selected)
    
    def add_custom_voice_to_selection(self, voice_path, selected_voices):
        """添加自定义声音到选择列表"""
        return add_custom_voice_to_selection(voice_path, selected_voices)
    
    def process_audio(self, input_file, selected_voices, token, steps, length, cfg, f0, auto_f0, pitch):
        """
        处理音频函数，带有错误处理功能
        
        参数:
            input_file: 输入音频文件路径
            selected_voices: 已选择的参考声音列表
            token: Hugging Face 认证令牌
            steps: 扩散步数
            length: 长度调整因子
            cfg: 推理 CFG 速率
            f0: 是否使用 F0 条件
            auto_f0: 是否自动调整 F0
            pitch: 音高调整（半音）
            
        返回:
            tuple: (输出音频路径, 状态消息)
        """
        # 参数检查
        is_valid, error_msg = validate_audio_params(input_file, selected_voices, token)
        if not is_valid:
            return None, error_msg
        
        try:
            # 解析音高调整值，如果有的话
            parsed_pitch = parse_pitch_shifts(pitch) if pitch else []
            
            # 调用匿名化处理函数，确保它能捕获所有异常
            try:
                return self.audio_anonymizer.anonymize(
                    input_file, 
                    selected_voices, 
                    token, 
                    diffusion_steps=steps, 
                    length_adjust=length, 
                    inference_cfg_rate=cfg, 
                    f0_condition=f0,
                    use_auto_f0_adjust=auto_f0,
                    pitch_shifts=parsed_pitch,
                    progress=gr.Progress()
                )
            except ValueError as e:
                return None, f"处理错误: {str(e)}"
            except Exception as e:
                return None, f"发生未预期的错误: {str(e)}，请检查参数或联系开发者"
        except Exception as e:
            return None, f"参数错误: {str(e)}"
    
    def _create_custom_show_voices(self, selected_voices_state, reference_voices):
        """
        创建自定义声音显示函数
        
        参数:
            selected_voices_state: 选择状态
            reference_voices: 参考声音组件
            
        返回:
            function: 显示自定义声音的函数
        """
        # 创建一个包装函数
        def wrapper_show_custom_voices():
            def on_add_callback(voice_path):
                # 调用工具函数，添加声音到选择列表
                updated, labels = add_custom_voice_to_selection(voice_path, selected_voices_state.value)
                selected_voices_state.value = updated
                reference_voices.choices = labels
                return updated, labels
            
            def on_delete_callback(voice_path):
                # 调用工具函数，删除声音并刷新
                status_msg = delete_reference_voice(voice_path)[0]
                # 更新下拉列表
                new_options = get_reference_voices_dropdown_options(self.reference_voices_dir)
                
                # 如果被删除的声音在已选择列表中，则移除它
                updated_selected = [v for v in selected_voices_state.value 
                                  if v != voice_path]
                
                # 创建新标签
                labels = []
                for vp in updated_selected:
                    voice_name = os.path.basename(vp)
                    if os.path.basename(vp).startswith("custom_"):
                        voice_name = os.path.basename(vp).replace("custom_", "")
                    labels.append((voice_name, vp))
                    
                selected_voices_state.value = updated_selected
                reference_voices.choices = labels
                
                return status_msg
            
            # 使用工具函数
            return show_custom_voices(
                self.reference_voices_dir, 
                on_add_callback=on_add_callback, 
                on_delete_callback=on_delete_callback
            )
            
        return wrapper_show_custom_voices
    
    def create_tab_ui(self):
        """
        创建可以嵌入到主应用标签页的音频处理UI组件
        
        返回:
            tuple: UI组件和回调函数
        """
        # 存储选择的参考声音列表
        selected_voices_state = gr.State([])
        
        tab_content = gr.Markdown("""
        # 音频匿名化工具
        
        此工具结合了 pyannote 和 Seed-VC，可以对音频中的多个说话人进行声音替换，实现音频脱敏。
        
        ## 使用流程:
        1. 上传需要处理的音频文件
        2. 选择或上传参考声音文件 (至少与音频中说话人数量相同)
        3. 设置处理参数
        4. 点击"开始处理"按钮
        """)
        
        with gr.Tabs(elem_id="audio_tab", selected="basic_settings") as tabs:
            with gr.Tab(label="基本设置", id="basic_settings"):
                with gr.Row():
                    with gr.Column():
                        input_audio = gr.Audio(type="filepath", label="输入音频文件")
                        auth_token = gr.Textbox(
                            label="Hugging Face 认证令牌", 
                            placeholder="hf_...", 
                            info="用于下载 pyannote 模型，需要在 Hugging Face 上申请"
                        )
                        auth_token.change()
                    
                    # 参考声音选择
                    with gr.Column():
                        # 获取参考声音下拉选项
                        dropdown_options = get_reference_voices_dropdown_options(
                            reference_voices_dir=self.reference_voices_dir
                        )
                        
                        # 创建参考声音下拉列表 - 直接作为选择的来源
                        reference_voices = gr.Dropdown(
                            choices=dropdown_options,
                            label="选择参考声音",
                            info="选择要用于声音替换的参考声音",
                            type="value",
                            multiselect=True,
                        )
                        
                        # 直接将reference_voices的值绑定到selected_voices_state
                        reference_voices.change(
                            fn=lambda x: x,  # 直接传递值
                            inputs=[reference_voices],
                            outputs=[selected_voices_state]
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
                        
                        # 创建上传回调函数
                        def upload_and_refresh(audio, name):
                            if not audio:
                                return None, "请先选择一个音频文件", ""
                            
                            result = upload_reference_voice(audio, self.reference_voices_dir, name)
                            updated_dropdown = get_reference_voices_dropdown_options(
                                reference_voices_dir=self.reference_voices_dir
                            )
                            return result[0], result[1], "", updated_dropdown
                        
                        upload_button.click(
                            fn=upload_and_refresh,
                            inputs=[custom_reference, custom_reference_name],
                            outputs=[custom_reference, upload_status, custom_reference_name, reference_voices]
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
                        info="&lt;1.0 加速语速，&gt;1.0 减慢语速"
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
                        # 使用工具函数创建默认声音显示
                        def add_default_to_selection(voice_path):
                            updated, labels = add_to_selection(
                                selected_voices_state.value, 
                                voice_path, 
                                dropdown_options
                            )
                            selected_voices_state.value = updated
                            reference_voices.choices = labels
                            return updated, labels
                        
                        # 获取默认参考声音列表并创建显示组件
                        default_voices = get_reference_voices(
                            reference_voices_dir=self.reference_voices_dir,
                            include_custom=False,
                        )
                        create_voice_group(default_voices, is_default=True, 
                                         on_add_callback=add_default_to_selection)
                    
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
                        
                        # 创建自定义声音动态组件
                        custom_voices_container = gr.Column(elem_id="custom_voices_container")
                        
                        # 创建自定义声音显示函数
                        show_custom_voices_fn = self._create_custom_show_voices(
                            selected_voices_state, 
                            reference_voices
                        )
                        
                        # 上传新参考声音的回调
                        def upload_and_refresh_all(audio, name):
                            if not audio:
                                return None, "请先选择一个音频文件", "", None
                            
                            result = upload_reference_voice(audio, self.reference_voices_dir, name)
                            # 刷新下拉列表和自定义声音显示
                            updated_dropdown = self.get_reference_voices_dropdown_options()
                            return result[0], result[1], "", updated_dropdown
                        
                        new_upload_button.click(
                            fn=upload_and_refresh_all,
                            inputs=[new_custom_reference, new_custom_name],
                            outputs=[new_custom_reference, new_upload_status, new_custom_name, reference_voices]
                        )
                        
                        # 初始化展示自定义声音
                        with custom_voices_container:
                            custom_components = show_custom_voices_fn()
                        
                        # 刷新自定义声音列表按钮
                        refresh_custom_btn = gr.Button("刷新自定义声音列表")
                        
                        # 刷新自定义声音列表回调
                        refresh_custom_btn.click(
                            fn=show_custom_voices_fn,
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
        
        # 使用类方法替代局部函数
        start_button.click(
            fn=self.process_audio,
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
        
        # 返回所有组件和回调函数
        return {
            "tab_content": tab_content,
            "input_audio": input_audio,
            "output_audio": output_audio,
            "speaker_info": speaker_info,
            "start_button": start_button
        }
    
    def initialize_gradio_ui(self):
        """
        创建 Gradio 用户界面
        
        返回:
            gr.Blocks: Gradio 界面
        """
        with gr.Blocks(title="音频匿名化工具") as demo:
            # 创建组件
            components = self.create_tab_ui()
            
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
