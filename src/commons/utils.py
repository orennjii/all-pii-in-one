# -*- coding: utf-8 -*-
"""
通用工具函数模块，包含多个项目组件共用的工具函数
"""

# System modules
import os
import glob
import tempfile
import shutil
from pathlib import Path

# External modules
import numpy as np
import yaml

# 内部模块
from .loggers import get_module_logger

# 获取当前模块的日志记录器
logger = get_module_logger(__name__)


def create_temp_dir(prefix="temp_"):
    """
    创建临时目录
    
    Args:
        prefix (str): 临时目录前缀
    Returns:
        str: 临时目录路径
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    logger.debug(f"创建临时目录: {temp_dir}")
    return temp_dir


def cleanup_temp_dir(temp_dir):
    """
    清理临时目录

    Args:
        temp_dir (str): 临时目录路径
    """
    if temp_dir and os.path.exists(temp_dir):
        logger.debug(f"清理临时目录: {temp_dir}")
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"临时目录清理完成: {temp_dir}")
        except Exception as e:
            logger.error(f"清理临时目录失败: {temp_dir}, 错误: {str(e)}")


def load_yaml_config(config_path):
    """
    加载YAML配置文件

    Args:
        config_path (str): 配置文件路径

    Returns:
        dict: 配置数据
    """
    logger.debug(f"加载配置文件: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_data = yaml.safe_load(f)
            logger.debug(f"成功加载配置文件: {config_path}")
            return config_data
    except Exception as e:
        logger.error(f"加载配置文件失败: {config_path}, 错误: {str(e)}")
        raise


def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory (str): 目录路径
    """
    try:
        if not os.path.exists(directory):
            logger.debug(f"创建目录: {directory}")
            os.makedirs(directory, exist_ok=True)
        else:
            logger.debug(f"目录已存在: {directory}")
    except Exception as e:
        logger.error(f"创建目录失败: {directory}, 错误: {str(e)}")
        raise

def find_project_root(
        current_path: Path | str,
        marker_filename: str = ".git"
    ) -> Path:
    """
    向上查找包含特定标记文件的项目根目录。
    """
    path = Path(current_path).resolve()

    while path != path.parent:
        if (path / marker_filename).exists():
            return path
        path = path.parent

    if (path / marker_filename).exists():
        return path
    else:
        raise FileNotFoundError(
            f"未找到包含标记文件 '{marker_filename}' 的项目根目录。"
        )


