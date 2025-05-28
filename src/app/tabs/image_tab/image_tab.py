#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理 Gradio UI 模块

提供与 main.py 相同功能的 Web 界面
使用 Presidio Image Redactor 进行图像中的PII检测和编辑
"""

import gradio as gr
from gradio.themes import Soft
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import tempfile
import time
import io
import logging
from contextlib import redirect_stdout, redirect_stderr

from PIL import Image
from presidio_image_redactor import ImageRedactorEngine, ImageAnalyzerEngine, TesseractOCR

from src.processors.text_processor.core.analyzer import PresidioAnalyzer
from src.configs import AppConfig
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)


class LogCapture:
    """日志捕获器，用于捕获处理过程中的日志信息"""
    
    def __init__(self):
        self.logs = []
        self.handler = None
        
    def start_capture(self):
        """开始捕获日志"""
        self.logs.clear()
        
        # 创建自定义处理器
        self.handler = logging.StreamHandler()
        self.handler.setLevel(logging.INFO)
        
        # 设置格式器
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.handler.setFormatter(formatter)
        
        # 添加到根日志记录器
        logging.getLogger().addHandler(self.handler)
        
    def stop_capture(self):
        """停止捕获日志"""
        if self.handler:
            logging.getLogger().removeHandler(self.handler)
            self.handler = None
            
    def get_logs(self) -> str:
        """获取捕获的日志"""
        return "\n".join(self.logs)


class ImageProcessorTab:
    """图像处理器 Gradio Tab"""
    
    def __init__(self):
        """初始化图像处理器 Tab"""
        self.image_analyzer = None
        self.image_engine = None
        self.project_root = find_project_root(Path(__file__))
        self.log_capture = LogCapture()
        self._load_config()
        
    def _load_config(self):
        """加载配置并初始化图像处理器"""
        try:
            config_path = self.project_root / "config" / "app_config.yaml"
            app_config = AppConfig.load_from_yaml(str(config_path))
            
            # 初始化 Presidio 分析器
            analyzer = PresidioAnalyzer(app_config.processor.text_processor)._analyzer
            logger.info(f"使用识别器: {analyzer.get_recognizers(language='zh')}")
            
            # 初始化 OCR
            ocr = TesseractOCR()
            
            # 初始化图像分析器
            self.image_analyzer = ImageAnalyzerEngine(
                analyzer_engine=analyzer,
                ocr=ocr,
            )
            logger.info("图像分析器初始化成功，使用 Presidio 分析器和 Tesseract OCR")
            
            # 初始化图像编辑器
            self.image_engine = ImageRedactorEngine(
                image_analyzer_engine=self.image_analyzer,
            )
            logger.info("图像处理器初始化成功")
            
        except Exception as e:
            logger.error(f"图像处理器初始化失败: {e}")
            
    def process_image(
        self,
        image,
        redact_color_r: int,
        redact_color_g: int,
        redact_color_b: int,
        ocr_language: str,
        language: str,
        output_filename: str
    ) -> Tuple[str, str, str, str]:
        """
        处理图像并返回结果
        
        Args:
            image: 上传的图像
            redact_color_r: 编辑颜色的红色分量
            redact_color_g: 编辑颜色的绿色分量
            redact_color_b: 编辑颜色的蓝色分量
            ocr_language: OCR识别语言
            language: PII检测语言
            output_filename: 输出文件名
            
        Returns:
            Tuple[编辑后图像路径, 处理日志, 统计信息, 处理状态]
        """
        if not self.image_engine or not self.image_analyzer:
            error_msg = "错误：图像处理器未初始化"
            return "", error_msg, error_msg, error_msg
            
        if image is None:
            error_msg = "请上传图像文件"
            return "", error_msg, error_msg, error_msg
            
        try:
            start_time = time.time()
            
            # 开始捕获日志
            log_stream = io.StringIO()
            
            # 创建临时日志处理器
            temp_handler = logging.StreamHandler(log_stream)
            temp_handler.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            temp_handler.setFormatter(formatter)
            
            # 添加处理器到相关日志记录器
            loggers_to_capture = [
                logging.getLogger(__name__),
                logging.getLogger('src.processors.text_processor'),
                logging.getLogger('presidio_image_redactor'),
                logging.getLogger('presidio_analyzer')
            ]
            
            for log in loggers_to_capture:
                log.addHandler(temp_handler)
            
            try:
                logger.info("开始处理图像")
                
                # 打开图像
                if isinstance(image, str):
                    pil_image = Image.open(image)
                else:
                    pil_image = image
                
                logger.info(f"图像尺寸: {pil_image.size}")
                logger.info(f"图像模式: {pil_image.mode}")
                
                # 设置编辑颜色
                redact_color = (redact_color_r, redact_color_g, redact_color_b)
                logger.info(f"编辑颜色: RGB{redact_color}")
                
                # 执行OCR和PII检测
                logger.info("开始OCR识别和PII检测")
                
                # 执行图像编辑
                redacted_image = self.image_engine.redact(
                    image=pil_image,
                    fill=redact_color,
                    ocr_kwargs={"lang": ocr_language},
                    language=language
                )
                
                logger.info("图像编辑完成")
                
                # 保存编辑后的图像
                output_dir = self.project_root / "data" / "image" / "output"
                output_dir.mkdir(parents=True, exist_ok=True)
                
                if output_filename.strip():
                    output_filename = output_filename.strip()
                    if not output_filename.endswith(('.png', '.jpg', '.jpeg')):
                        output_filename += '.png'
                else:
                    output_filename = f"redacted_image_{int(time.time())}.png"
                
                output_path = output_dir / output_filename
                redacted_image.save(output_path)
                
                processing_time = time.time() - start_time
                
                logger.info(f"图像处理完成，用时 {processing_time:.2f} 秒")
                logger.info(f"编辑后图像保存至: {output_path}")
                
                # 获取捕获的日志
                log_content = log_stream.getvalue()
                
                # 生成统计信息
                statistics = self._format_statistics(
                    pil_image, 
                    redacted_image, 
                    processing_time, 
                    output_path,
                    redact_color,
                    ocr_language,
                    language
                )
                
                status_text = f"✅ 图像处理完成！用时 {processing_time:.2f} 秒"
                
                return str(output_path), log_content, statistics, status_text
                
            finally:
                # 移除临时处理器
                for log in loggers_to_capture:
                    log.removeHandler(temp_handler)
                temp_handler.close()
                
        except Exception as e:
            error_msg = f"处理失败: {str(e)}"
            logger.error(error_msg)
            return "", f"错误: {error_msg}", f"错误: {error_msg}", f"❌ {error_msg}"
    
    def _format_statistics(
        self, 
        original_image: Image.Image, 
        redacted_image: Image.Image, 
        processing_time: float,
        output_path: Path,
        redact_color: tuple,
        ocr_language: str,
        language: str
    ) -> str:
        """格式化统计信息"""
        stats_lines = [
            "=== 图像处理统计 ===",
            f"总处理时间: {processing_time:.2f} 秒",
            "",
            "原始图像信息:",
            f"  • 尺寸: {original_image.size[0]} x {original_image.size[1]} 像素",
            f"  • 模式: {original_image.mode}",
            f"  • 格式: {getattr(original_image, 'format', '未知')}",
            "",
            "编辑后图像信息:",
            f"  • 尺寸: {redacted_image.size[0]} x {redacted_image.size[1]} 像素",
            f"  • 模式: {redacted_image.mode}",
            f"  • 保存路径: {output_path.name}",
            "",
            "处理配置:",
            f"  • 编辑颜色: RGB{redact_color}",
            f"  • OCR语言: {ocr_language}",
            f"  • PII检测语言: {language}",
            "",
            "处理流程:",
            "  1. ✅ 图像加载",
            "  2. ✅ OCR文字识别",
            "  3. ✅ PII实体检测",
            "  4. ✅ 敏感信息编辑",
            "  5. ✅ 图像保存"
        ]
        
        return "\n".join(stats_lines)
    
    def get_example_image_info(self) -> str:
        """获取示例图像信息"""
        return """
### 📁 示例图像文件位置

项目中提供了一些测试图像文件，位于：
- `image.png` - 项目根目录下的示例图像
- `data/image/` 目录下的其他图像文件

### 🖼️ 支持的图像格式
- PNG, JPG, JPEG, BMP, TIFF
- 分辨率: 建议不超过4096x4096以获得最佳性能
- 文件大小: 建议不超过10MB

### 📋 处理功能
1. **OCR文字识别**: 使用Tesseract识别图像中的文字
2. **PII检测**: 在识别的文字中检测个人隐私信息
3. **敏感信息编辑**: 用指定颜色的矩形覆盖敏感信息区域
4. **多语言支持**: 支持中文、英文等多种语言的OCR和PII检测
        """
    
    def clear_all(self) -> Tuple[None, str, str, str, str]:
        """清空所有内容"""
        return None, "", "", "", "已清空所有内容"
    
    def create_interface(self) -> gr.Blocks:
        """创建 Gradio 界面"""
        with gr.Blocks(title="图像PII处理", theme=Soft()) as interface:
            gr.Markdown("# 🖼️ 图像隐私信息检测与编辑工具")
            gr.Markdown("基于 Presidio Image Redactor 的图像隐私信息处理工具，支持OCR识别、PII检测和敏感信息编辑。")
            
            with gr.Row():
                with gr.Column(scale=2):
                    # 输入区域
                    gr.Markdown("## 📥 图像输入")
                    image_input = gr.Image(
                        label="上传图像文件",
                        type="pil",
                        height=300
                    )
                    
                    # 配置区域
                    gr.Markdown("## ⚙️ 处理配置")
                    
                    with gr.Row():
                        redact_color_r = gr.Slider(
                            label="编辑颜色 - 红色",
                            minimum=0,
                            maximum=255,
                            value=0,
                            step=1
                        )
                        redact_color_g = gr.Slider(
                            label="编辑颜色 - 绿色", 
                            minimum=0,
                            maximum=255,
                            value=0,
                            step=1
                        )
                        redact_color_b = gr.Slider(
                            label="编辑颜色 - 蓝色",
                            minimum=0,
                            maximum=255,
                            value=0,
                            step=1
                        )
                    
                    with gr.Row():
                        ocr_language = gr.Dropdown(
                            label="OCR识别语言",
                            choices=["chi_sim", "eng", "chi_tra"],
                            value="chi_sim",
                            info="选择OCR文字识别语言"
                        )
                        
                        language = gr.Dropdown(
                            label="PII检测语言",
                            choices=["zh", "en"],
                            value="zh",
                            info="选择PII检测语言"
                        )
                    
                    output_filename = gr.Textbox(
                        label="输出文件名",
                        placeholder="redacted_image.png",
                        value="",
                        info="指定编辑后图像的文件名（可选）"
                    )
                    
                    with gr.Row():
                        process_btn = gr.Button("🚀 开始处理", variant="primary", size="lg")
                        clear_btn = gr.Button("🗑️ 清空", variant="secondary")
                
                with gr.Column(scale=1):
                    # 状态和说明
                    gr.Markdown("## 📖 使用说明")
                    with gr.Accordion("点击查看详细说明", open=False):
                        gr.Markdown(self.get_example_image_info())
                    
                    # 处理状态
                    status_output = gr.Textbox(
                        label="处理状态",
                        interactive=False,
                        lines=2
                    )
            
            # 输出区域
            gr.Markdown("## 📤 处理结果")
            
            with gr.Tabs():
                with gr.TabItem("🖼️ 编辑后图像"):
                    redacted_image_output = gr.Image(
                        label="编辑后图像",
                        interactive=False,
                        height=400
                    )
                
                with gr.TabItem("📊 统计信息"):
                    statistics_output = gr.Textbox(
                        label="处理统计",
                        lines=12,
                        max_lines=20,
                        interactive=False
                    )
                
                with gr.TabItem("📋 处理日志"):
                    log_output = gr.Textbox(
                        label="处理日志",
                        lines=15,
                        max_lines=25,
                        interactive=False,
                        placeholder="处理日志将在这里显示..."
                    )
            
            # 事件绑定
            process_btn.click(
                fn=self.process_image,
                inputs=[
                    image_input,
                    redact_color_r,
                    redact_color_g,
                    redact_color_b,
                    ocr_language,
                    language,
                    output_filename
                ],
                outputs=[
                    redacted_image_output,
                    log_output,
                    statistics_output,
                    status_output
                ]
            )
            
            clear_btn.click(
                fn=self.clear_all,
                outputs=[
                    image_input,
                    log_output,
                    statistics_output,
                    status_output,
                    redacted_image_output
                ]
            )
            
            # 添加详细说明
            with gr.Accordion("📋 功能说明", open=False):
                gr.Markdown("""
                ### 🔧 主要功能
                
                1. **OCR文字识别**: 使用 Tesseract OCR 从图像中提取文字内容
                2. **PII检测**: 使用 Presidio 分析器检测提取文字中的个人隐私信息
                3. **敏感信息编辑**: 在检测到PII的区域用指定颜色的矩形进行覆盖
                4. **多语言支持**: 支持中文、英文等多种语言的识别和检测
                
                ### 📝 使用流程
                
                1. **上传图像**: 选择包含文字的图像文件
                2. **配置选项**: 
                   - 设置编辑颜色（RGB值）
                   - 选择OCR识别语言
                   - 选择PII检测语言
                   - 指定输出文件名（可选）
                3. **开始处理**: 点击"开始处理"按钮，系统将：
                   - 使用OCR识别图像中的文字
                   - 检测文字中的PII实体
                   - 在敏感信息区域进行编辑
                   - 保存编辑后的图像
                4. **查看结果**: 在不同标签页中查看编辑后图像、统计信息和处理日志
                
                ### 🎨 编辑颜色设置
                
                - **黑色**: R=0, G=0, B=0（默认，适合大多数情况）
                - **白色**: R=255, G=255, B=255
                - **红色**: R=255, G=0, B=0
                - **模糊效果**: 可以使用接近背景的颜色
                
                ### 🌐 语言支持
                
                **OCR语言**:
                - `chi_sim`: 简体中文
                - `eng`: 英语
                - `chi_tra`: 繁体中文
                
                **PII检测语言**:
                - `zh`: 中文
                - `en`: 英文
                
                ### 📁 输出文件
                
                - 编辑后的图像会保存到 `data/image/output/` 目录
                - 如果未指定文件名，将自动生成带时间戳的文件名
                - 支持PNG、JPG等常见格式
                
                ### ⚡ 性能说明
                
                - **OCR识别**: 取决于图像大小和文字复杂度
                - **PII检测**: 通常在1-2秒内完成
                - **图像编辑**: 几乎实时完成
                - **建议图像尺寸**: 不超过4096x4096像素以获得最佳性能
                """)
        
        return interface


def create_image_tab() -> gr.Blocks:
    """创建图像处理标签页"""
    tab = ImageProcessorTab()
    return tab.create_interface()


if __name__ == "__main__":
    # 独立运行测试
    demo = create_image_tab()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7863,
        share=False,
        debug=True
    )
