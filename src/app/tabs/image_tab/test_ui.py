#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
图像处理 UI 测试脚本

测试图像处理 Gradio UI 的功能
PYTHONPATH=. python src/app/tabs/image_tab/test_ui.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from src.app.tabs.image_tab import create_image_tab
from src.commons import get_module_logger

logger = get_module_logger(__name__)

def test_image_ui():
    """测试图像处理UI"""
    logger.info("启动图像处理UI测试...")
    
    try:
        # 创建UI
        demo = create_image_tab()
        
        # 启动UI
        demo.launch(
            server_name="0.0.0.0",
            server_port=7863,
            share=False,
            debug=True
        )
        
    except Exception as e:
        logger.error(f"图像UI测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_image_ui()
