#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置使用示例
"""

from src.configs import AUDIO_CONFIG


def main():
    """
    展示如何使用音频配置
    """
    # 使用全局单例实例
    print("=== 使用全局配置单例示例 ===")
    print(f"说话人分割模型: {AUDIO_CONFIG.diarization.model}")
    print(f"最小分段时长: {AUDIO_CONFIG.diarization.min_segment_duration}秒")
    print(f"支持的音频格式: {', '.join(AUDIO_CONFIG.supported_formats.file_types)}")
    print(f"扩散步数: {AUDIO_CONFIG.voice_conversion.diffusion_steps}")
    
    # 不可变性示例
    print("\n=== 配置不可变性示例 ===")
    try:
        # 这里会抛出错误，因为配置是不可变的
        AUDIO_CONFIG.diarization.model = "new-model"
    except Exception as e:
        print(f"尝试修改配置失败: {e}")
    
    # 配置值不变
    print(f"说话人分割模型 (仍保持原值): {AUDIO_CONFIG.diarization.model}")


if __name__ == "__main__":
    main()
