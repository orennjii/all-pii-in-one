#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
提示词加载器模块

负责加载和格式化LLM提示词模板，支持多种模板格式和变量替换。
"""

import json
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.commons.loggers import get_module_logger

logger = get_module_logger(__name__)


class PromptTemplate:
    """
    提示词模板类
    
    封装单个提示词模板的信息和格式化功能。
    """
    
    def __init__(
        self, 
        name: str, 
        template: str, 
        description: str = "", 
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        初始化提示词模板
        
        Args:
            name: 模板名称
            template: 模板内容
            description: 模板描述
            metadata: 模板元数据
        """
        self.name = name
        self.template = template
        self.description = description
        self.metadata = metadata or {}
    
    def format(self, **kwargs) -> str:
        """
        格式化模板
        
        Args:
            **kwargs: 模板变量
            
        Returns:
            str: 格式化后的提示词
            
        Raises:
            KeyError: 当缺少必需的模板变量时
            ValueError: 当模板格式无效时
        """
        try:
            # 使用双花括号语法进行变量替换
            formatted_prompt = self.template
            
            for key, value in kwargs.items():
                placeholder = f"{{{{ {key} }}}}"
                formatted_prompt = formatted_prompt.replace(placeholder, str(value))
            
            # 检查是否还有未替换的占位符
            if "{{" in formatted_prompt and "}}" in formatted_prompt:
                import re
                unresolved = re.findall(r'\{\{\s*(\w+)\s*\}\}', formatted_prompt)
                if unresolved:
                    logger.warning(f"模板 '{self.name}' 中有未解析的变量: {unresolved}")
            
            return formatted_prompt
            
        except Exception as e:
            raise ValueError(f"格式化模板 '{self.name}' 失败: {str(e)}") from e
    
    def get_variables(self) -> List[str]:
        """
        获取模板中的所有变量名
        
        Returns:
            List[str]: 变量名列表
        """
        import re
        variables = re.findall(r'\{\{\s*(\w+)\s*\}\}', self.template)
        return list(set(variables))  # 去重
    
    def get_required_variables(self) -> List[str]:
        """
        获取模板中的必需变量名（别名方法，与get_variables相同）
        
        Returns:
            List[str]: 必需变量名列表
        """
        return self.get_variables()
    
    def validate_variables(self, **kwargs) -> List[str]:
        """
        验证提供的变量是否满足模板需求
        
        Args:
            **kwargs: 要验证的变量
            
        Returns:
            List[str]: 缺少的变量名列表
        """
        required_vars = self.get_variables()
        provided_vars = set(kwargs.keys())
        missing_vars = [var for var in required_vars if var not in provided_vars]
        return missing_vars
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PromptTemplate(name='{self.name}', vars={self.get_variables()})"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()


class PromptLoader:
    """
    提示词加载器
    
    负责从文件系统加载提示词模板，并提供模板管理和格式化功能。
    """
    
    def __init__(self, template_path: Optional[str] = None):
        """
        初始化提示词加载器
        
        Args:
            template_path: 模板文件路径，如果为None则使用默认路径
        """
        if template_path is None:
            # 使用默认路径 - 从当前文件位置向上查找项目根目录
            current_dir = Path(__file__).resolve()
            project_root = None
            
            # 向上查找直到找到包含config目录的父目录
            for parent in current_dir.parents:
                if (parent / "config").exists():
                    project_root = parent
                    break
            
            if project_root is None:
                # 如果找不到，使用相对路径
                project_root = Path.cwd()
                
            template_path = os.path.join(project_root, "config", "prompt_template.json")
        
        self.template_path = Path(template_path)
        self.templates: Dict[str, PromptTemplate] = {}
        self._load_templates()
    
    def _load_templates(self) -> None:
        """从文件加载模板"""
        try:
            if not self.template_path.exists():
                logger.warning(f"模板文件不存在: {self.template_path}")
                return
            
            with open(self.template_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if not isinstance(data, dict):
                raise ValueError("模板文件格式无效：根对象必须是字典")
            
            for name, template_data in data.items():
                if isinstance(template_data, dict):
                    # 新格式：包含描述和元数据
                    template = PromptTemplate(
                        name=name,
                        template=template_data.get("template", ""),
                        description=template_data.get("description", ""),
                        metadata=template_data.get("metadata", {})
                    )
                elif isinstance(template_data, str):
                    # 旧格式：只有模板内容
                    template = PromptTemplate(
                        name=name,
                        template=template_data
                    )
                else:
                    logger.warning(f"跳过无效的模板格式: {name}")
                    continue
                
                self.templates[name] = template
                logger.debug(f"加载模板: {name}")
            
            logger.info(f"成功加载 {len(self.templates)} 个提示词模板")
            
        except FileNotFoundError:
            logger.error(f"模板文件未找到: {self.template_path}")
        except json.JSONDecodeError as e:
            logger.error(f"模板文件JSON格式无效: {str(e)}")
        except Exception as e:
            logger.error(f"加载模板文件失败: {str(e)}")
    
    def reload_templates(self) -> None:
        """重新加载模板"""
        self.templates.clear()
        self._load_templates()
    
    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """
        获取指定名称的模板
        
        Args:
            name: 模板名称
            
        Returns:
            Optional[PromptTemplate]: 模板对象，不存在时返回None
        """
        return self.templates.get(name)
    
    def has_template(self, name: str) -> bool:
        """
        检查模板是否存在
        
        Args:
            name: 模板名称
            
        Returns:
            bool: 如果模板存在返回True，否则返回False
        """
        return name in self.templates
    
    def list_templates(self) -> List[str]:
        """
        获取所有模板名称
        
        Returns:
            List[str]: 模板名称列表
        """
        return list(self.templates.keys())
    
    def load_prompt_template(self, name: str) -> PromptTemplate:
        """
        加载指定名称的提示词模板
        
        Args:
            name: 模板名称
            
        Returns:
            PromptTemplate: 提示词模板对象
            
        Raises:
            KeyError: 当模板不存在时
        """
        if name not in self.templates:
            available_templates = list(self.templates.keys())
            raise KeyError(
                f"未找到模板 '{name}'。"
                f"可用的模板: {available_templates}"
            )
        
        return self.templates[name]
    
    def format_prompt(self, template_name: str, **kwargs) -> str:
        """
        格式化指定模板
        
        Args:
            template_name: 模板名称
            **kwargs: 模板变量
            
        Returns:
            str: 格式化后的提示词
            
        Raises:
            KeyError: 当模板不存在时
            ValueError: 当模板格式化失败时
        """
        template = self.load_prompt_template(template_name)
        
        # 验证变量
        missing_vars = template.validate_variables(**kwargs)
        if missing_vars:
            logger.warning(
                f"模板 '{template_name}' 缺少变量: {missing_vars}。"
                f"需要的变量: {template.get_variables()}"
            )
        
        return template.format(**kwargs)
    
    def get_template_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        获取模板信息
        
        Args:
            name: 模板名称
            
        Returns:
            Optional[Dict[str, Any]]: 模板信息字典，不存在时返回None
        """
        template = self.get_template(name)
        if not template:
            return None
        
        return {
            "name": template.name,
            "description": template.description,
            "variables": template.get_variables(),
            "metadata": template.metadata
        }
    
    def search_templates(self, keyword: str) -> List[str]:
        """
        根据关键词搜索模板
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            List[str]: 匹配的模板名称列表
        """
        keyword_lower = keyword.lower()
        matching_templates = []
        
        for name, template in self.templates.items():
            if (keyword_lower in name.lower() or 
                keyword_lower in template.description.lower()):
                matching_templates.append(name)
        
        return matching_templates
    
    def add_template(
        self, 
        name: str, 
        template: str, 
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        save_to_file: bool = False
    ) -> None:
        """
        添加新模板
        
        Args:
            name: 模板名称
            template: 模板内容
            description: 模板描述
            metadata: 模板元数据
            save_to_file: 是否保存到文件
        """
        if name in self.templates:
            logger.warning(f"模板 '{name}' 已存在，将被覆盖")
        
        self.templates[name] = PromptTemplate(
            name=name,
            template=template,
            description=description,
            metadata=metadata
        )
        
        logger.info(f"添加模板: {name}")
        
        if save_to_file:
            self._save_templates()
    
    def remove_template(self, name: str, save_to_file: bool = False) -> bool:
        """
        删除模板
        
        Args:
            name: 模板名称
            save_to_file: 是否保存到文件
            
        Returns:
            bool: 如果删除成功返回True，否则返回False
        """
        if name in self.templates:
            del self.templates[name]
            logger.info(f"删除模板: {name}")
            
            if save_to_file:
                self._save_templates()
            
            return True
        
        return False
    
    def _save_templates(self) -> None:
        """保存模板到文件"""
        try:
            # 确保目录存在
            self.template_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 构建保存数据
            data = {}
            for name, template in self.templates.items():
                data[name] = {
                    "template": template.template,
                    "description": template.description,
                    "metadata": template.metadata
                }
            
            # 保存到文件
            with open(self.template_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"模板已保存到: {self.template_path}")
            
        except Exception as e:
            logger.error(f"保存模板文件失败: {str(e)}")
    
    def __len__(self) -> int:
        """返回模板数量"""
        return len(self.templates)
    
    def __contains__(self, name: str) -> bool:
        """检查模板是否存在"""
        return name in self.templates
    
    def __iter__(self):
        """迭代器支持"""
        return iter(self.templates.values())
    
    def __str__(self) -> str:
        """字符串表示"""
        return f"PromptLoader(templates={len(self.templates)}, path='{self.template_path}')"
    
    def __repr__(self) -> str:
        """详细字符串表示"""
        return self.__str__()
