"""
音频匿名化器测试脚本
用于测试AudioAnonymizer类的功能
"""

import os
import sys
import traceback


# 从AudioAnonymizer导入相关类
from src.processors.audio_processor.audio_anonymizer import AudioAnonymizer



def test_audio_anonymizer():
    """
    测试AudioAnonymizer类的基本功能
    """
    try:
        # 手动指定参数
        input_path = os.path.join(ROOT_DIR, "data/audio/source_voices/source_s1.wav")  # 输入音频路径
        output_path = os.path.join(ROOT_DIR, "data/audio/output/anonymized_s1.wav")    # 输出音频路径
        reference_voice = "s2p1"                                                       # 参考声音
        device = "cpu"                                                                 # 设备
        min_speakers = 1                                                               # 最少说话人数
        max_speakers = 3                                                               # 最多说话人数
        enable_pii_detection = False                                                   # 是否启用PII检测
        
        print(f"项目根目录: {ROOT_DIR}")
        print(f"输入文件: {input_path}")
        print(f"输出文件: {output_path}")
        
        # 确保输入文件存在
        if not os.path.exists(input_path):
            print(f"错误: 输入音频文件不存在: {input_path}")
            return 1
            
        # 初始化音频匿名化器
        print(f"初始化音频匿名化器，使用设备: {device}")
        anonymizer = AudioAnonymizer(
            device=device,
            enable_pii_detection=enable_pii_detection
        )
        
        # 显示可用的参考声音
        print("可用的参考声音:")
        for name in anonymizer.reference_voices.keys():
            print(f"  - {name}")
        
        if len(anonymizer.reference_voices) == 0:
            print("警告: 没有找到可用的参考声音")
            
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            print(f"创建输出目录: {output_dir}")
            
        # 执行音频匿名化
        print(f"开始处理音频文件...")
        if enable_pii_detection:
            output_path, pii_result = anonymizer.anonymize(
                audio_path=input_path,
                output_path=output_path,
                reference_voice=reference_voice,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
                return_pii_info=True
            )
            
            # 显示PII检测结果
            print("\nPII检测结果:")
            if pii_result and pii_result.get('pii_found', False):
                print(f"在音频中检测到 {len(pii_result.get('segments', []))} 个PII片段")
                for i, segment in enumerate(pii_result.get('segments', [])):
                    print(f"  片段 {i+1}: {segment.get('text', '')} ({segment.get('start', 0):.2f}s - {segment.get('end', 0):.2f}s)")
                    print(f"    检测到的PII类型: {', '.join(segment.get('pii_types', []))}")
            else:
                print("未检测到PII信息")
        else:
            output_path = anonymizer.anonymize(
                audio_path=input_path,
                output_path=output_path,
                reference_voice=reference_voice,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
        
        if os.path.exists(output_path):
            print(f"\n处理成功! 匿名化音频已保存至: {output_path}")
            print(f"文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        else:
            print(f"\n错误: 输出文件未生成: {output_path}")
            return 1
            
    except Exception as e:
        print(f"错误: {str(e)}")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    # 执行测试
    result = test_audio_anonymizer()
    sys.exit(result)
