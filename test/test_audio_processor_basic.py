#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
AudioProcessor 完整流程测试

测试音频处理器的完整功能：转录 → PII检测 → 音频匿名化
可以通过以下方式运行:
PYTHONPATH=. python test/test_audio_processor_basic.py
"""

import os
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from src.configs import AppConfig
from src.processors.audio_processor import AudioProcessor
from src.processors.audio_processor.core.pii_detection import AudioPIIResult, PIIEntity
from src.commons import get_module_logger, find_project_root

logger = get_module_logger(__name__)

current_path = Path(__file__).resolve()
project_root = find_project_root(current_path)

def test_complete_audio_processing():
    """测试完整的音频处理流程"""
    
    logger.info("=" * 60)
    logger.info("AudioProcessor 完整流程测试")
    logger.info("=" * 60)
    
    # 加载配置并初始化处理器
    config_path = project_root / "config" / "app_config.yaml"
    app_config = AppConfig.load_from_yaml(str(config_path))
    audio_processor = AudioProcessor(config=app_config)
    
    # 使用实际音频文件
    audio_path = project_root / "data" / "audio" / "source_voices" / "test_audio2.mp3"
    output_dir = project_root / "data" / "audio" / "output"
    
    logger.info(f"测试音频文件: {audio_path}")
    
    # 检查文件是否存在
    if not audio_path.exists():
        logger.error(f"❌ 音频文件不存在: {audio_path}")
        return
    
    try:
        # 步骤4: 完整流程测试（使用主要的process方法）
        logger.info("\n" + "=" * 40)
        logger.info("步骤4: 完整流程测试")
        logger.info("=" * 40)
        
        try:
            # 使用 AudioProcessor 的主要处理方法
            complete_result = audio_processor.process(
                audio_path=str(audio_path),
                output_dir=str(output_dir)
            )
            
            if complete_result:
                logger.info(f"✅ 完整流程处理成功!")
                logger.info(f"   原始音频: {complete_result.original_audio_path}")
                if complete_result.anonymized_audio_path:
                    logger.info(f"   匿名化音频: {complete_result.anonymized_audio_path}")
                    
                    # 验证匿名化音频
                    try:
                        import librosa
                        anon_audio_data, anon_sr = librosa.load(complete_result.anonymized_audio_path, sr=None)
                        logger.info(f"   匿名化音频时长: {len(anon_audio_data)/anon_sr:.2f}秒")
                        logger.info(f"   匿名化音频采样率: {anon_sr} Hz")
                    except ImportError:
                        logger.info("   (无法验证匿名化音频详情，librosa未安装)")
                        
                if complete_result.pii_detection_result:
                    logger.info(f"   检测到PII实体: {len(complete_result.pii_detection_result.pii_entities)} 个")
                    
                    # 显示完整流程检测到的PII实体
                    for i, entity in enumerate(complete_result.pii_detection_result.pii_entities):
                        logger.info(f"     实体 {i+1}: {entity.entity_type} = '{entity.text}' "
                                   f"({entity.start_time:.1f}s-{entity.end_time:.1f}s)")
                        
                if complete_result.transcription_result:
                    logger.info(f"   转录片段数: {len(complete_result.transcription_result.get('segments', []))}")
                    
                if complete_result.diarization_result:
                    logger.info(f"   说话人分离结果: 已生成")
                else:
                    logger.info("   说话人分离: 未启用或未检测到多说话人")
            else:
                logger.warning("⚠️ 完整流程处理返回None")
                
        except Exception as e:
            logger.warning(f"完整流程测试失败: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info("\n" + "=" * 60)
        logger.info("AudioProcessor 完整流程测试完成!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_complete_audio_processing()
