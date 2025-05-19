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


def create_temp_dir():
    """
    创建临时目录
    
    返回:
        str: 临时目录路径
    """
    return tempfile.mkdtemp()


def cleanup_temp_dir(temp_dir):
    """
    清理临时目录
    
    参数:
        temp_dir: 临时目录路径
    """
    if temp_dir and os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)


def load_yaml_config(config_path):
    """
    加载YAML配置文件
    
    参数:
        config_path: 配置文件路径
        
    返回:
        dict: 配置数据
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def ensure_directory_exists(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory: 目录路径
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


