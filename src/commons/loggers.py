#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
日志记录模块 (Logger Module)

本模块提供项目范围内统一的日志记录功能，支持多种日志配置和管理方式。
主要特点：
- 单例模式实现的日志管理器，确保全局唯一日志配置
- 支持基于大小的日志轮转 (RotatingFileHandler)
- 支持基于时间的日志轮转 (TimedRotatingFileHandler)
- 支持模块级别的日志记录
- 支持控制台和文件同时输出
- 提供便捷的全局日志记录函数

基本用法：
1. 初始化日志系统:
   ```python
   from commons.loggers import init_logging
   init_logging(log_dir="/path/to/logs", level=logging.INFO)
   ```

2. 获取默认日志记录器:
   ```python
   from commons.loggers import get_default_logger, info, error
   
   # 使用函数式API
   info("这是一条信息")
   error("发生错误", exc_info=True)
   
   # 使用Logger对象
   logger = get_default_logger()
   logger.info("这是一条信息")
   ```

3. 获取基于模块的日志记录器:
   ```python
   from commons.loggers import get_module_logger
   
   # 自动使用当前模块名称
   logger = get_module_logger()
   logger.info("模块日志记录")
   
   # 或指定模块名称
   logger = get_module_logger("custom.module")
   ```

4. 创建自定义日志记录器:
   ```python
   from commons.loggers import get_logger, get_time_rotating_logger
   
   # 普通日志记录器
   logger = get_logger("my_logger", level=logging.DEBUG, log_file="custom.log")
   
   # 基于时间轮转的日志记录器
   timed_logger = get_time_rotating_logger("daily_logger", when="D", interval=1)
   ```

完整示例见 scripts/logger_example.py
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# 默认日志格式
DEFAULT_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
# 简化的日志格式（用于控制台输出）
SIMPLE_FORMAT = "%(levelname)s: %(message)s"


class LoggerManager:
    """
    日志管理器类
    用于创建和管理项目中的所有日志记录器
    """
    
    # 单例模式
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._loggers = {}
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, log_dir=None, level=logging.INFO, add_console=True):
        """
        初始化日志管理器
        
        Args:
            log_dir: 日志文件保存目录，默认为项目根目录下的logs文件夹
            level: 默认日志级别
            add_console: 是否添加控制台处理器
        """
        if self._initialized:
            return
            
        # 设置日志目录
        if log_dir is None:
            # 获取项目根目录
            project_root = Path(__file__).parent.parent.parent
            log_dir = project_root / "logs"
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.level = level
        self.add_console = add_console
        self._initialized = True
        
        # 创建默认日志记录器
        self._default_logger = self.get_logger("default")
    
    def get_logger(
        self,
        name,
        level=None,
        log_file=None,
        format_str=None, 
        max_bytes=10*1024*1024, backup_count=5, add_console=None
    ):
        """
        获取或创建一个日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别, 默认使用LoggerManager的level
            log_file: 日志文件名, 默认为name.log
            format_str: 日志格式
            max_bytes: 单个日志文件最大大小（字节）
            backup_count: 备份日志文件数量
            add_console: 是否添加控制台处理器，默认使用LoggerManager的设置
            
        Returns:
            logging.Logger: 配置好的日志记录器对象
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # 设置默认值
        if level is None:
            level = self.level
        if log_file is None:
            log_file = f"{name}.log"
        if format_str is None:
            format_str = DEFAULT_FORMAT
        if add_console is None:
            add_console = self.add_console
            
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 防止重复添加处理器
        if logger.handlers:
            return logger
            
        # 配置文件处理器
        log_path = self.log_dir / log_file
        file_handler = RotatingFileHandler(
            log_path, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8'
        )
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 配置控制台处理器
        if add_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(SIMPLE_FORMAT)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        # 保存日志记录器
        self._loggers[name] = logger
        return logger
        
    def get_time_rotating_logger(self, name, level=None, log_file=None, format_str=None,
                                 when='D', interval=1, backup_count=30, add_console=None):
        """
        获取或创建一个基于时间轮转的日志记录器
        
        Args:
            name: 日志记录器名称
            level: 日志级别，默认使用LoggerManager的level
            log_file: 日志文件名，默认为name.log
            format_str: 日志格式
            when: 轮转周期，'S':秒, 'M':分, 'H':小时, 'D':天, 'W':周
            interval: 轮转间隔
            backup_count: 备份日志文件数量
            add_console: 是否添加控制台处理器，默认使用LoggerManager的设置
            
        Returns:
            logging.Logger: 配置好的日志记录器对象
        """
        if name in self._loggers:
            return self._loggers[name]
        
        # 设置默认值
        if level is None:
            level = self.level
        if log_file is None:
            log_file = f"{name}.log"
        if format_str is None:
            format_str = DEFAULT_FORMAT
        if add_console is None:
            add_console = self.add_console
            
        # 创建日志记录器
        logger = logging.getLogger(name)
        logger.setLevel(level)
        
        # 防止重复添加处理器
        if logger.handlers:
            return logger
            
        # 配置时间轮转文件处理器
        log_path = self.log_dir / log_file
        file_handler = TimedRotatingFileHandler(
            log_path, when=when, interval=interval, 
            backupCount=backup_count, encoding='utf-8'
        )
        file_formatter = logging.Formatter(format_str)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 配置控制台处理器
        if add_console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = logging.Formatter(SIMPLE_FORMAT)
            console_handler.setFormatter(console_formatter)
            logger.addHandler(console_handler)
            
        # 保存日志记录器
        self._loggers[name] = logger
        return logger
    
    def get_module_logger(self, module_name, add_console=None):
        """
        获取模块级别的日志记录器，使用模块名称作为日志记录器名称
        
        Args:
            module_name: 模块名称，通常传入__name__
            add_console: 是否添加控制台处理器
            
        Returns:
            logging.Logger: 配置好的日志记录器对象
        """
        return self.get_logger(module_name, add_console=add_console)
        
    @property
    def default_logger(self):
        """获取默认日志记录器"""
        return self._default_logger


# 全局日志管理器单例
_logger_manager = None

def init_logging(log_dir=None, level=logging.INFO, add_console=True):
    """
    初始化全局日志管理器
    
    Args:
        log_dir: 日志文件保存目录
        level: 默认日志级别
        add_console: 是否添加控制台处理器
        
    Returns:
        LoggerManager: 日志管理器实例
    """
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager(log_dir, level, add_console)
    return _logger_manager

def get_logger(name, **kwargs):
    """
    获取一个日志记录器，如果全局日志管理器未初始化则先初始化
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他参数传递给LoggerManager.get_logger方法
        
    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    if _logger_manager is None:
        init_logging()
    return _logger_manager.get_logger(name, **kwargs)

def get_time_rotating_logger(name, **kwargs):
    """
    获取一个基于时间轮转的日志记录器
    
    Args:
        name: 日志记录器名称
        **kwargs: 其他参数传递给LoggerManager.get_time_rotating_logger方法
        
    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    if _logger_manager is None:
        init_logging()
    return _logger_manager.get_time_rotating_logger(name, **kwargs)

def get_module_logger(module_name=None, **kwargs):
    """
    获取模块级别的日志记录器，使用调用者的模块名作为日志记录器名称
    
    Args:
        module_name: 模块名称，默认为调用者的__name__
        **kwargs: 其他参数传递给LoggerManager.get_module_logger方法
        
    Returns:
        logging.Logger: 配置好的日志记录器对象
    """
    if _logger_manager is None:
        init_logging()
        
    if module_name is None:
        # 尝试获取调用者的模块名
        import inspect
        frm = inspect.stack()[1]
        module_name = inspect.getmodule(frm[0]).__name__
        
    return _logger_manager.get_module_logger(module_name, **kwargs)

# 默认日志记录器
def get_default_logger():
    """获取默认日志记录器"""
    if _logger_manager is None:
        init_logging()
    return _logger_manager.default_logger

# 便捷函数，直接通过默认日志记录器记录日志
def debug(msg, *args, **kwargs):
    get_default_logger().debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    get_default_logger().info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    get_default_logger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    get_default_logger().error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    get_default_logger().critical(msg, *args, **kwargs)

def exception(msg, *args, **kwargs):
    get_default_logger().exception(msg, *args, **kwargs)
