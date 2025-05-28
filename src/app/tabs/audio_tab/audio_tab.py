#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
音频处理 Gradio UI 模块

提供与 test_audio_processor_basic.py 相同功能的 Web 界面
"""

import gradio as gr
from gradio.themes import Soft
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import json
import tempfile
import time

from src.configs import AppConfig
from src.processors.audio_processor import AudioProcessor
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)


class AudioProcessorTab:
    """音频处理器 Gradio Tab"""
    
    def __init__(self):
        """初始化音频处理器 Tab"""
        self.audio_processor = None
        self.project_root = find_project_root(Path(__file__))
        self._load_config()
        
    def _load_config(self):
        """加载配置并初始化音频处理器"""
        try:
            config_path = self.project_root / "config" / "app_config.yaml"
            app_config = AppConfig.load_from_yaml(str(config_path))
            self.audio_processor = AudioProcessor(config=app_config)
            logger.info("音频处理器初始化成功")
        except Exception as e:
            logger.error(f"音频处理器初始化失败: {e}")
            
    def process_audio(
        self,
        audio_file,
        enable_diarization: bool,
        output_dir_name: str
    ) -> Tuple[str, str, str, str, str, str]:
        """
        处理音频并返回结果
        
        Args:
            audio_file: 上传的音频文件
            enable_diarization: 是否启用说话人分离
            output_dir_name: 输出目录名称
            
        Returns:
            Tuple[转录结果, 说话人分离结果, PII检测结果, 统计信息, 匿名化音频路径, 处理状态]
        """
        if not self.audio_processor:
            error_msg = "错误：音频处理器未初始化"
            return error_msg, error_msg, error_msg, error_msg, "", error_msg
            
        if not audio_file:
            error_msg = "请上传音频文件"
            return error_msg, error_msg, error_msg, error_msg, "", error_msg
            
        try:
            start_time = time.time()
            
            # 设置输出目录
            output_dir = None
            if output_dir_name.strip():
                output_dir = self.project_root / "data" / "audio" / "output" / output_dir_name.strip()
                output_dir.mkdir(parents=True, exist_ok=True)
            
            # 执行完整音频处理流程
            logger.info(f"开始处理音频文件: {audio_file}")
            
            processing_result = self.audio_processor.process(
                audio_path=audio_file,
                enable_diarization=enable_diarization,
                output_dir=output_dir
            )
            
            processing_time = time.time() - start_time
            
            # 格式化结果
            transcription_text = self._format_transcription_results(processing_result.transcription_result)
            diarization_text = self._format_diarization_results(processing_result.diarization_result)
            pii_detection_text = self._format_pii_results(processing_result.pii_detection_result)
            statistics_text = self._format_statistics(processing_result, processing_time)
            
            # 匿名化音频路径
            anonymized_audio_path = processing_result.anonymized_audio_path or ""
            
            status_text = f"✅ 音频处理完成！用时 {processing_time:.2f} 秒"
            
            return (
                transcription_text,
                diarization_text, 
                pii_detection_text,
                statistics_text,
                anonymized_audio_path,
                status_text
            )
            
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logger.error(error_msg)
            return error_msg, error_msg, error_msg, error_msg, "", f"❌ {error_msg}"
    
    def _format_transcription_results(self, transcription_result) -> str:
        """格式化转录结果"""
        if not transcription_result:
            return "未获得转录结果"
        
        result_lines = ["=== 语音转录结果 ===\n"]
        
        # 检测到的语言
        language = transcription_result.get('language', 'unknown')
        result_lines.append(f"检测语言: {language}\n")
        
        # 转录片段
        segments = transcription_result.get('segments', [])
        if segments:
            result_lines.append(f"转录片段数: {len(segments)}\n")
            result_lines.append("=== 转录内容 ===")
            
            for i, segment in enumerate(segments):
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                text = segment.get('text', '').strip()
                
                result_lines.append(
                    f"[{start_time:.1f}s - {end_time:.1f}s] {text}"
                )
        else:
            result_lines.append("未检测到转录片段")
        
        return "\n".join(result_lines)
    
    def _format_diarization_results(self, diarization_result) -> str:
        """格式化说话人分离结果"""
        if not diarization_result:
            return "说话人分离未启用或未检测到多个说话人"
        
        result_lines = ["=== 说话人分离结果 ===\n"]
        
        # 说话人数量
        speaker_count = getattr(diarization_result, 'speaker_count', 0)
        result_lines.append(f"检测到说话人数: {speaker_count}\n")
        
        # 说话人片段
        speaker_segments = getattr(diarization_result, 'speaker_segments', [])
        if speaker_segments:
            result_lines.append("=== 说话人片段 ===")
            
            for segment in speaker_segments:
                speaker = segment.get('speaker', 'unknown')
                start_time = segment.get('start', 0)
                end_time = segment.get('end', 0)
                
                result_lines.append(
                    f"说话人 {speaker}: [{start_time:.1f}s - {end_time:.1f}s]"
                )
        else:
            result_lines.append("未检测到说话人片段详情")
        
        return "\n".join(result_lines)
    
    def _format_pii_results(self, pii_result) -> str:
        """格式化PII检测结果"""
        if not pii_result or not pii_result.pii_entities:
            return "未检测到任何 PII 实体"
        
        result_lines = ["=== PII 检测结果 ===\n"]
        
        total_entities = len(pii_result.pii_entities)
        result_lines.append(f"检测到 PII 实体总数: {total_entities}\n")
        
        result_lines.append("=== PII 实体详情 ===")
        
        for i, entity in enumerate(pii_result.pii_entities):
            result_lines.append(
                f"实体 {i+1}:"
                f"  • 类型: {entity.entity_type}"
                f"  • 文本: '{entity.text}'"
                f"  • 时间: {entity.start_time:.1f}s - {entity.end_time:.1f}s"
                f"  • 置信度: {entity.confidence:.2f}"
            )
            if hasattr(entity, 'speaker_id') and entity.speaker_id:
                result_lines.append(f"  • 说话人: {entity.speaker_id}")
            result_lines.append("")
        
        return "\n".join(result_lines)
    
    def _format_statistics(self, processing_result, processing_time: float) -> str:
        """格式化统计信息"""
        stats_lines = [
            "=== 处理统计 ===",
            f"总处理时间: {processing_time:.2f} 秒",
            f"原始音频: {Path(processing_result.original_audio_path).name}",
            ""
        ]
        
        # 转录统计
        if processing_result.transcription_result:
            segments = processing_result.transcription_result.get('segments', [])
            total_duration = 0
            total_text_length = 0
            
            for segment in segments:
                duration = segment.get('end', 0) - segment.get('start', 0)
                total_duration += duration
                total_text_length += len(segment.get('text', ''))
            
            stats_lines.extend([
                "转录统计:",
                f"  • 音频总时长: {total_duration:.1f} 秒",
                f"  • 转录片段数: {len(segments)}",
                f"  • 转录文本总长度: {total_text_length} 字符",
                f"  • 检测语言: {processing_result.transcription_result.get('language', 'unknown')}",
                ""
            ])
        
        # 说话人分离统计
        if processing_result.diarization_result:
            speaker_count = getattr(processing_result.diarization_result, 'speaker_count', 0)
            stats_lines.extend([
                "说话人分离统计:",
                f"  • 说话人数量: {speaker_count}",
                ""
            ])
        
        # PII检测统计
        if processing_result.pii_detection_result and processing_result.pii_detection_result.pii_entities:
            pii_entities = processing_result.pii_detection_result.pii_entities
            
            # 按类型统计
            entity_types = {}
            for entity in pii_entities:
                entity_types[entity.entity_type] = entity_types.get(entity.entity_type, 0) + 1
            
            stats_lines.extend([
                "PII检测统计:",
                f"  • PII实体总数: {len(pii_entities)}",
            ])
            
            for entity_type, count in sorted(entity_types.items()):
                stats_lines.append(f"  • {entity_type}: {count} 个")
            
            stats_lines.append("")
        
        # 匿名化统计
        if processing_result.anonymized_audio_path:
            stats_lines.extend([
                "匿名化统计:",
                f"  • 匿名化音频: {Path(processing_result.anonymized_audio_path).name}",
                "  • 状态: 已完成",
                ""
            ])
        else:
            stats_lines.extend([
                "匿名化统计:",
                "  • 状态: 未执行（无PII实体检测到）",
                ""
            ])
        
        return "\n".join(stats_lines)
    
    def get_example_audio_info(self) -> str:
        """获取示例音频信息"""
        return """
### 📁 示例音频文件位置

项目中提供了一些测试音频文件，位于：
- `data/audio/source_voices/test_audio2.mp3`
- `data/audio/source_voices/` 目录下的其他音频文件

### 🎵 支持的音频格式
- MP3, WAV, FLAC, M4A, OGG
- 采样率: 16kHz 或更高（推荐）
- 时长: 建议不超过10分钟以获得最佳性能

### 📋 处理功能
1. **语音转录**: 将音频转换为文字
2. **说话人分离**: 识别不同的说话人（可选）
3. **PII检测**: 在转录文本中检测个人隐私信息
4. **音频匿名化**: 用蜂鸣声替换包含PII的音频片段
        """
    
    def clear_all(self) -> Tuple[None, str, str, str, str, str, str]:
        """清空所有内容"""
        return None, "", "", "", "", "", "已清空所有内容"
    
    def create_interface(self) -> gr.Blocks:
        """创建 Gradio 界面"""
        with gr.Blocks(title="音频PII处理", theme=Soft()) as interface:
            gr.Markdown("# 🎵 音频隐私信息检测与匿名化工具")
            gr.Markdown("基于 AudioProcessor 的音频隐私信息处理工具，支持语音转录、说话人分离、PII检测和音频匿名化。")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    gr.Markdown("## 📥 音频输入")
                    audio_input = gr.Audio(
                        label="上传音频文件",
                        type="filepath",
                        format="wav"
                    )
                    
                    # 配置区域
                    gr.Markdown("## ⚙️ 处理配置")
                    
                    enable_diarization = gr.Checkbox(
                        label="启用说话人分离",
                        value=False,
                        info="识别音频中的不同说话人（处理时间较长）"
                    )
                    
                    output_dir_input = gr.Textbox(
                        label="输出目录名称",
                        placeholder="anonymized_audio",
                        value="",
                        info="指定匿名化音频的输出目录名（可选）"
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                with gr.Column(scale=1):
                    # 状态和说明
                    gr.Markdown("## 📖 使用说明")
                    with gr.Accordion("点击查看详细说明", open=False):
                        gr.Markdown(self.get_example_audio_info())
                    
                    # 处理状态
                    status_output = gr.Textbox(
                        label="处理状态",
                        interactive=False,
                        lines=2
                    )
            
            # 输出区域
            gr.Markdown("## 📤 处理结果")
            
            with gr.Tabs():
                with gr.TabItem("🎙️ 转录结果"):
                    transcription_output = gr.Textbox(
                        label="语音转录结果",
                        lines=12,
                        max_lines=20,
                        interactive=False
                    )
                
                with gr.TabItem("👥 说话人分离"):
                    diarization_output = gr.Textbox(
                        label="说话人分离结果",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                
                with gr.TabItem("🔍 PII检测"):
                    pii_output = gr.Textbox(
                        label="PII检测结果",
                        lines=12,
                        max_lines=20,
                        interactive=False
                    )
                
                with gr.TabItem("📊 统计信息"):
                    statistics_output = gr.Textbox(
                        label="处理统计",
                        lines=10,
                        max_lines=15,
                        interactive=False
                    )
                
                with gr.TabItem("🔇 匿名化音频"):
                    gr.Markdown("### 匿名化后的音频文件")
                    anonymized_audio_output = gr.Audio(
                        label="匿名化音频",
                        interactive=False
                    )
                    anonymized_path_output = gr.Textbox(
                        label="匿名化文件路径",
                        interactive=False,
                        lines=1
                    )
            
            # 事件绑定
            process_btn.click(
                fn=self.process_audio,
                inputs=[
                    audio_input,
                    enable_diarization,
                    output_dir_input
                ],
                outputs=[
                    transcription_output,
                    diarization_output,
                    pii_output,
                    statistics_output,
                    anonymized_path_output,
                    status_output
                ]
            )
            
            # 当匿名化路径更新时，自动加载音频
            anonymized_path_output.change(
                fn=lambda path: path if path and Path(path).exists() else None,
                inputs=anonymized_path_output,
                outputs=anonymized_audio_output
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[
                    audio_input,
                    transcription_output,
                    diarization_output,
                    pii_output,
                    statistics_output,
                    anonymized_path_output,
                    status_output
                ]
            )
            
            # 添加详细说明
            with gr.Accordion("📋 功能说明", open=False):
                gr.Markdown("""
                ### 🔧 主要功能
                
                1. **语音转录**: 使用 WhisperX 模型将音频转换为文字，支持多种语言自动检测
                2. **说话人分离**: 使用 Pyannote 模型识别音频中的不同说话人
                3. **PII检测**: 在转录文本中检测以下类型的个人隐私信息：
                   - 人名、电话号码、邮箱地址
                   - 身份证号、银行账号、信用卡号
                   - 地理位置、组织机构、日期时间等
                4. **音频匿名化**: 对包含PII的音频片段用蜂鸣声进行替换
                
                ### 📝 使用流程
                
                1. **上传音频**: 选择要处理的音频文件
                2. **配置选项**: 
                   - 选择是否启用说话人分离（会增加处理时间）
                   - 指定输出目录名称（可选）
                3. **开始处理**: 点击"开始处理"按钮，系统将：
                   - 自动转录音频内容
                   - 如果启用，进行说话人分离
                   - 检测转录文本中的PII实体
                   - 如果检测到PII，生成匿名化音频
                4. **查看结果**: 在不同标签页中查看各项处理结果
                
                ### ⚡ 性能说明
                
                - **转录**: 通常需要 0.1-0.3 倍的音频时长
                - **说话人分离**: 额外增加 0.2-0.5 倍的音频时长
                - **PII检测**: 几秒钟内完成
                - **音频匿名化**: 取决于检测到的PII数量
                
                ### 📁 输出文件
                
                - 匿名化音频会保存到指定的输出目录
                - 如果未指定目录，将保存到默认位置
                - 可以在"匿名化音频"标签页中直接播放和下载
                """)
        
        return interface


def create_audio_tab() -> gr.Blocks:
    """创建音频处理标签页"""
    tab = AudioProcessorTab()
    return tab.create_interface()


if __name__ == "__main__":
    # 独立运行测试
    demo = create_audio_tab()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,
        share=False,
        debug=True
    )