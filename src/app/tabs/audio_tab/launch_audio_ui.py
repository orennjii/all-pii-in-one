# System modules
import os
import argparse
from pathlib import Path

# External modules
import torch
import gradio as gr

# Internal modules
from src.modules.seed_vc_modules.commons import str2bool
from src.processors.audio_processor.processor import AudioAnonymizer
from .audio_ui import AudioUI
from src.commons import (
    get_device,
    get_specific_cuda_device,
    find_project_root,
)


def main():
    """
    主函数，解析命令行参数并启动应用
    """
    # 设置参考声音目录
    current_path = Path(__file__)
    reference_voices_dir = find_project_root(current_path) / "data" / "audio" / "reference_voices"

    # 定义命令行参数
    parser = argparse.ArgumentParser(description="音频匿名化工具 - Web 界面")
    parser.add_argument("--share", type=str2bool, nargs="?", const=True, default=False, help="是否共享应用")
    parser.add_argument("--gpu", type=int, help="使用的 GPU ID", default=0)
    args = parser.parse_args()
    
    # 设置设备
    device = get_specific_cuda_device(args.gpu) if torch.cuda.is_available() else get_device()
    print(f"使用设备: {device}")
    
    # 初始化用户界面
    audio_ui = AudioUI(
        device=torch.device('cpu'),
        reference_voices_dir=reference_voices_dir,
    )
    
    # 使用修改后的接口创建并启动 Gradio 界面
    demo = audio_ui.initialize_gradio_ui()
    demo.launch(share=args.share)

if __name__ == "__main__":
    main()
