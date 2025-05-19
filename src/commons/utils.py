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


def create_temp_dir(prefix="temp_"):
    """
    创建临时目录
    
    Args:
        prefix (str): 临时目录前缀
    Returns:
        str: 临时目录路径
    """
    return tempfile.mkdtemp(prefix=prefix)


def cleanup_temp_dir(temp_dir):
    """
    清理临时目录

    Args:
        temp_dir (str): 临时目录路径
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def load_yaml_config(config_path):
    """
    加载YAML配置文件

    Args:
        config_path (str): 配置文件路径

    Returns:
        dict: 配置数据
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    Args:
        directory (str): 目录路径
    """
    os.makedirs(directory, exist_ok=True)

def find_project_root(
        current_path: Path | str,
        marker_filename: str = ".git"
    ) -> Path | None:
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
    return None


