#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
文本处理 UI 测试脚本

测试文本处理 Gradio UI 的功能
"""

import sys
import os
from pathlib import Path

# 添加项目根目录到 Python 路径
current_path = Path(__file__).resolve()
project_root = current_path.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.tabs.text_tab import create_text_tab
from src.commons import get_module_logger

logger = get_module_logger(__name__)

def test_text_tab():
    """测试文本处理 Tab"""
    logger.info("开始测试文本处理 UI...")
    
    try:
        # 创建 UI
        demo = create_text_tab()
        logger.info("文本处理 UI 创建成功")
        
        # 启动 UI
        logger.info("启动文本处理 UI，访问地址: http://localhost:7862")
        demo.launch(
            server_name="0.0.0.0",
            server_port=7862,
            share=False,
            debug=True,
            show_error=True
        )
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        raise

if __name__ == "__main__":
    test_text_tab()
