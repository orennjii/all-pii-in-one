#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
配置系统使用示例
"""

import os
import sys

# 添加项目根目录到 Python 路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from src.configs import AppConfig

# 构建配置文件的绝对路径
config_path = os.path.join(project_root, "config", "app_config.yaml")

# 从配置文件加载配置
config = AppConfig.load_from_yaml(config_path)

# 也可以使用单例模式获取配置实例
# config = AppConfig.get_instance(config_path)

# 访问嵌套配置
print(f"日志级别: {config.general.log_level}")
print(f"UI端口: {config.general.ui.server_port}")
print(f"使用GPU: {config.device.use_gpu}")
print(f"模式识别器启用状态: {config.text_processor.recognizers.pattern.enabled}")
print(f"LLM识别器启用状态: {config.text_processor.recognizers.llm.enabled}")
print(f"LLM模型: {config.text_processor.recognizers.llm.client.model_name_or_path}")

# 配置对象是不可变的（frozen=True），尝试修改会引发错误
try:
    config.general.log_level = "DEBUG"
except Exception as e:
    print(f"尝试修改配置时发生错误: {e}")

# 使用 update 方法创建更新后的配置
update_dict = {
    "general": {
        "log_level": "DEBUG",
        "ui": {
            "server_port": 8080
        }
    }
}
updated_config = config.update(update_dict)
print(f"更新后的日志级别: {updated_config.general.log_level}")
print(f"更新后的UI端口: {updated_config.general.ui.server_port}")
print(f"更新后的模式识别器状态（保持不变）: {updated_config.text_processor.recognizers.pattern.enabled}")

# 以前的方式创建新配置（不推荐，因为会丢失未指定的配置）
partial_config = AppConfig.from_dict(update_dict)
print(f"\n使用 from_dict 直接创建的配置:")
print(f"日志级别: {partial_config.general.log_level}")
print(f"UI端口: {partial_config.general.ui.server_port}")
# 注意这里会使用默认值而不是原配置中的值
print(f"模式识别器状态（使用默认值）: {partial_config.text_processor.recognizers.pattern.enabled}")
