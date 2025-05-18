#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
设备配置模块，用于获取和管理设备类型（GPU、MPS或CPU）
此模块是通用的，被多个专用模块使用
"""

import torch


def get_device():
    """
    根据可用硬件选择设备（CUDA、MPS 或 CPU）
    
    返回:
        torch.device: 可用的设备
    """
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    
    return torch.device(device)


def get_specific_cuda_device(gpu_id=0):
    """
    获取特定的CUDA设备
    
    参数:
        gpu_id: CUDA设备ID
        
    返回:
        torch.device: 指定的CUDA设备，如果不可用则返回默认设备
    """
    cuda_target = f"cuda:{gpu_id}" if torch.cuda.is_available() and gpu_id is not None else "cuda"
    return torch.device(cuda_target)