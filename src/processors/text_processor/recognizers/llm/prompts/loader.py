#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提示词加载器模块

用于加载和管理LLM提示词模板。
"""

import os
import json
import jinja2
from typing import Dict, Any, Optional

from src.commons.loggers import get_module_logger

logger = get_module_logger(__name__)


class PromptLoader:
    """
    提示词模板加载器
    
    用于从文件或字符串加载和渲染提示词模板。
    支持JSON格式的模板文件，可包含多个命名模板。
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        初始化提示词加载器
        
        Args:
            template_path: 模板文件路径，可选
        """
        self.template_path = template_path
        self.templates = {}
        
        # 如果提供了模板路径，则加载模板
        if template_path:
            self.load_templates(template_path)
            
        # 初始化Jinja2环境
        self.jinja_env = jinja2.Environment(
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True
        )
    
    def load_templates(self, template_path: str) -> Dict[str, str]:
        """
        从文件加载提示词模板
        
        Args:
            template_path: 模板文件路径
            
        Returns:
            Dict[str, str]: 模板字典，键为模板名，值为模板内容
            
        Raises:
            FileNotFoundError: 如果模板文件不存在
            ValueError: 如果模板文件格式错误
        """
        if not os.path.exists(template_path):
            # 如果路径不存在，尝试查找相对于项目根目录的路径
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
            full_path = os.path.join(base_dir, template_path)
            
            if not os.path.exists(full_path):
                logger.error(f"提示词模板文件不存在: {template_path}")
                raise FileNotFoundError(f"模板文件不存在: {template_path}")
            
            template_path = full_path
        
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                templates = json.load(f)
            
            if not isinstance(templates, dict):
                logger.error(f"提示词模板文件格式错误: {template_path}")
                raise ValueError(f"模板文件必须是JSON对象: {template_path}")
            
            self.templates = templates
            self.template_path = template_path
            logger.debug(f"已加载{len(templates)}个提示词模板")
            return templates
            
        except json.JSONDecodeError as e:
            logger.error(f"提示词模板文件解析错误: {str(e)}")
            raise ValueError(f"模板文件必须是有效的JSON格式: {str(e)}")
    
    def load_prompt_template(self, template_name: str) -> str:
        """
        获取指定名称的提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            str: 模板内容
            
        Raises:
            KeyError: 如果模板不存在
        """
        if not self.templates:
            # 如果未加载模板，尝试加载默认模板
            if self.template_path:
                self.load_templates(self.template_path)
            else:
                # 使用内置的默认模板
                return self.get_default_template(template_name)
        
        if template_name not in self.templates:
            logger.warning(f"提示词模板'{template_name}'不存在，将使用默认模板")
            return self.get_default_template(template_name)
        
        return self.templates[template_name]
    
    def render_template(self, template_name: str, **kwargs) -> str:
        """
        渲染提示词模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            str: 渲染后的提示词
        """
        template_content = self.load_prompt_template(template_name)
        template = self.jinja_env.from_string(template_content)
        return template.render(**kwargs)
    
    def get_default_template(self, template_name: str) -> str:
        """
        获取默认提示词模板
        
        Args:
            template_name: 模板名称
            
        Returns:
            str: 默认模板内容
        """
        default_templates = {
            "pii_detection": """请识别下面文本中可能的个人身份信息 (PII)，识别的实体类型包括: {entities}
            
文本内容：
{text}

请以 JSON 格式返回识别结果，格式如下:
[
  {
    "entity_type": "实体类型",
    "value": "实体值",
    "start": 开始位置(数字),
    "end": 结束位置(数字),
    "confidence": 置信度(0-1之间的浮点数)
  },
  ...
]

只返回 JSON，不要有其他内容。""",
        
            "pii_extraction": """请从以下文本中提取所有的个人隐私信息:

{text}

请列出所有个人隐私信息，包括但不限于:
- 姓名
- 身份证号
- 电话号码
- 邮箱地址
- 银行卡号
- 地址

按照以下JSON格式返回结果:
[
  {
    "type": "信息类型",
    "value": "具体值",
    "position": [开始位置, 结束位置]
  },
  ...
]
"""
        }
        
        if template_name not in default_templates:
            logger.error(f"默认提示词模板'{template_name}'不存在")
            raise KeyError(f"默认模板'{template_name}'不存在")
        
        return default_templates[template_name]
