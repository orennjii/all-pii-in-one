#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PII处理应用主入口

整合文本和音频处理的 Gradio 应用界面
"""

import gradio as gr
from pathlib import Path
import sys

# 添加项目根目录到 Python 路径
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.tabs.text_tab import create_text_tab
from src.commons import get_module_logger

logger = get_module_logger(__name__)


def create_main_app():
    """创建主应用界面"""
    
    with gr.Blocks(
        title="PII隐私信息处理工具",
        theme=gr.themes.Soft(),
        css="""
        .header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .tab-content {
            padding: 1rem;
        }
        """
    ) as app:
        
        # 应用标题和描述
        with gr.Row(elem_classes="header"):
            gr.HTML("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h1>🔒 PII隐私信息处理工具</h1>
                <p style="font-size: 1.1em; color: #666;">
                    基于AI的个人隐私信息检测与匿名化平台，支持文本和音频处理
                </p>
            </div>
            """)
        
        # 标签页
        with gr.Tabs():
            with gr.TabItem("📝 文本处理", elem_classes="tab-content"):
                text_interface = create_text_tab()
            
            with gr.TabItem("🎵 音频处理", elem_classes="tab-content"):
                gr.Markdown("""
                ## 🚧 音频处理功能
                
                音频处理功能正在开发中，将包括：
                - 语音转录
                - 说话人分离
                - 音频中的PII检测
                - 语音匿名化
                
                敬请期待！
                """)
            
            with gr.TabItem("📊 批量处理", elem_classes="tab-content"):
                gr.Markdown("""
                ## 📊 批量处理功能
                
                批量处理功能将支持：
                - 批量文本文件处理
                - 批量音频文件处理
                - 处理结果导出
                - 进度跟踪
                
                敬请期待！
                """)
            
            with gr.TabItem("⚙️ 设置", elem_classes="tab-content"):
                gr.Markdown("""
                ## ⚙️ 系统设置
                
                设置功能将包括：
                - 处理器配置
                - 模型选择
                - 性能调优
                - 日志查看
                
                敬请期待！
                """)
    
    return app


def main():
    """主函数"""
    logger.info("启动PII处理应用...")
    
    try:
        app = create_main_app()
        logger.info("应用创建成功")
        
        # 启动应用
        logger.info("启动应用服务器...")
        app.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=True,
            show_error=True,
            favicon_path=None,
            app_kwargs={
                "docs_url": None,
                "redoc_url": None,
            }
        )
        
    except Exception as e:
        logger.error(f"应用启动失败: {e}")
        raise


if __name__ == "__main__":
    main()
