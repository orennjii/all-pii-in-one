#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini模型响应解析器模块

该模块提供用于解析Google Gemini模型响应的具体实现。
处理Gemini返回的JSON、表格和纯文本格式，将其转换为标准的PII识别结果格式。
"""

import re
import json
from typing import Dict, Any, Optional, List, Tuple, Union

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm.parsers_config import LLMParsersConfig
from .parser import ResponseParser

logger = get_module_logger(__name__)


class GeminiResponseParser(ResponseParser):
    """
    Google Gemini API响应解析器
    
    专门解析来自Google Gemini模型的响应，支持多种输出格式：
    - JSON格式：标准结构化格式
    - 表格格式：Markdown或ASCII表格
    - 文本格式：自然语言描述
    """
    
    def __init__(self, config: Optional[LLMParsersConfig] = None, **kwargs):
        """
        初始化Gemini解析器
        
        Args:
            config: LLM解析器配置
            **kwargs: 其他参数，可包括:
                expected_format: 期望的响应格式，可选值：'json', 'table', 'text', 'auto'
                fallback_to_text: 如果结构化解析失败，是否回退到文本解析
                strict_mode: 是否使用严格模式解析JSON
                default_score: 默认置信度分数
        """
        # 使用配置
        if not config:
            from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
            config = LLMParsersConfig()
            
        # 合并配置和kwargs
        kwargs['fallback_to_text'] = kwargs.get('fallback_to_text', True)
        kwargs['default_score'] = kwargs.get('default_score', 0.8)
        kwargs['strict_mode'] = kwargs.get('strict_mode', config.json_strict_mode)
        
        # 期望格式和支持的格式
        self.expected_format = kwargs.get('expected_format', 'auto')
        self.supported_formats = config.gemini_response_formats
        
        super().__init__(**kwargs)
        
        logger.debug(f"Gemini解析器已初始化, 期望格式: {self.expected_format}")
    
    def parse(
        self, 
        llm_response: str,
        original_text: str, 
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析Gemini响应并返回识别结果列表
        
        Args:
            llm_response: LLM返回的原始文本响应
            original_text: 原始输入文本
            entities: 要查找的实体类型列表（可选）
            
        Returns:
            List[Dict[str, Any]]: 包含实体识别结果的列表，每个结果包含:
                - entity_type: 实体类型
                - start: 在原始文本中的起始位置
                - end: 在原始文本中的结束位置
                - confidence: 置信度分数
                - value: 实体值
        """
        if not llm_response or not original_text:
            logger.warning("收到空响应或原始文本为空")
            return []
            
        # 尝试不同格式的解析，按优先级排序
        if self.expected_format == 'auto':
            # 自动检测格式
            formats_to_try = ['json', 'table', 'text']
        else:
            # 使用指定格式，如果失败则回退到文本格式
            formats_to_try = [self.expected_format]
            if self.fallback_to_text and self.expected_format != 'text':
                formats_to_try.append('text')
                
        results = []
        
        for fmt in formats_to_try:
            if fmt not in self.supported_formats:
                continue
                
            try:
                logger.debug(f"尝试以{fmt}格式解析Gemini响应")
                if fmt == 'json':
                    results = self._parse_json_response(llm_response, original_text)
                elif fmt == 'table':
                    results = self._parse_table_response(llm_response, original_text)
                elif fmt == 'text':
                    results = self._parse_text_response(llm_response, original_text)
                    
                if results:
                    logger.debug(f"成功以{fmt}格式解析Gemini响应，找到{len(results)}个实体")
                    break
            except Exception as e:
                logger.warning(f"{fmt}格式解析失败: {str(e)}")
                continue
        
        # 过滤实体（如果指定）
        if entities and results:
            results = [r for r in results if r.get('entity_type', '').upper() in [e.upper() for e in entities]]
            
        return results
    
    def _parse_json_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """
        解析JSON格式的响应
        
        Args:
            response: Gemini响应文本
            original_text: 原始文本
            
        Returns:
            List[Dict[str, Any]]: 解析结果列表
        """
        # 提取JSON部分
        json_data = self._extract_json_from_text(response)
        if not json_data:
            raise ValueError("响应不包含有效的JSON数据")
            
        # 处理不同的JSON结构
        results = []
        
        # 处理数组格式
        if isinstance(json_data, list):
            for item in json_data:
                result = self._process_json_entity(item, original_text)
                if result:
                    results.append(result)
        # 处理包含entities字段的对象格式
        elif isinstance(json_data, dict):
            if 'entities' in json_data and isinstance(json_data['entities'], list):
                for item in json_data['entities']:
                    result = self._process_json_entity(item, original_text)
                    if result:
                        results.append(result)
            # 处理包含results字段的对象格式
            elif 'results' in json_data and isinstance(json_data['results'], list):
                for item in json_data['results']:
                    result = self._process_json_entity(item, original_text)
                    if result:
                        results.append(result)
            # 处理直接是实体列表的对象
            else:
                # 可能对象本身就是一个实体
                result = self._process_json_entity(json_data, original_text)
                if result:
                    results.append(result)
                    
        return results
        
    def _process_json_entity(self, entity_data: Dict[str, Any], original_text: str) -> Optional[Dict[str, Any]]:
        """
        处理单个JSON实体数据
        
        Args:
            entity_data: 实体JSON数据
            original_text: 原始文本
            
        Returns:
            Optional[Dict[str, Any]]: 处理后的实体数据，如果无效则返回None
        """
        # 检查必要的字段
        entity_type = entity_data.get('type') or entity_data.get('entity_type') or entity_data.get('entityType')
        if not entity_type:
            return None
            
        # 规范化实体类型
        entity_type = self._normalize_entity_type(entity_type)
        
        # 获取实体值
        entity_value = entity_data.get('value') or entity_data.get('text') or entity_data.get('content')
        if not entity_value:
            return None
            
        # 获取位置信息
        start = entity_data.get('start')
        end = entity_data.get('end')
        
        # 如果没有提供位置信息，尝试自己查找
        if start is None or end is None:
            start, end = self._find_text_position(original_text, entity_value)
            if start == -1:  # 找不到位置
                return None
                
        # 获取置信度，如果没有则使用默认值
        confidence = entity_data.get('confidence') or entity_data.get('score') or self.default_score
        
        return {
            'entity_type': entity_type,
            'start': start,
            'end': end,
            'confidence': float(confidence),
            'value': entity_value
        }
        
    def _parse_table_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """
        解析表格格式的响应（Markdown表格）
        
        Args:
            response: Gemini响应文本
            original_text: 原始文本
            
        Returns:
            List[Dict[str, Any]]: 解析结果列表
        """
        results = []
        
        # 尝试匹配Markdown表格
        table_pattern = r'\|([^|]+)\|([^|]+)\|([^|]+)\|([^|]*)\|'
        headers = None
        
        for line in response.split('\n'):
            line = line.strip()
            
            # 跳过表头分隔符行 (| --- | --- | --- |)
            if re.match(r'\|\s*[-:]+\s*\|', line):
                continue
                
            match = re.match(table_pattern, line)
            if match:
                if not headers:
                    # 这是表头行，记录列名
                    headers = [h.strip().lower() for h in match.groups()]
                    continue
                    
                # 这是数据行
                columns = [col.strip() for col in match.groups()]
                
                # 解析表格数据
                entity = {}
                for i, header in enumerate(headers):
                    if i < len(columns):
                        if 'type' in header or 'entity' in header:
                            entity['entity_type'] = columns[i]
                        elif 'value' in header or 'text' in header or 'content' in header:
                            entity['value'] = columns[i]
                        elif 'confidence' in header or 'score' in header:
                            try:
                                entity['confidence'] = float(columns[i])
                            except ValueError:
                                entity['confidence'] = self.default_score
                                
                # 如果表格包含了必要的信息
                if 'entity_type' in entity and 'value' in entity:
                    # 规范化实体类型
                    entity['entity_type'] = self._normalize_entity_type(entity['entity_type'])
                    
                    # 查找位置
                    start, end = self._find_text_position(original_text, entity['value'])
                    if start != -1:
                        entity['start'] = start
                        entity['end'] = end
                        if 'confidence' not in entity:
                            entity['confidence'] = self.default_score
                        results.append(entity)
        
        return results
        
    def _parse_text_response(self, response: str, original_text: str) -> List[Dict[str, Any]]:
        """
        解析文本格式的响应
        
        Args:
            response: Gemini响应文本
            original_text: 原始文本
            
        Returns:
            List[Dict[str, Any]]: 解析结果列表
        """
        results = []
        
        # 尝试使用正则表达式匹配实体描述
        # 匹配格式如: "Type/Entity: Person, Value: John Doe" 或 "Person: John Doe"
        entity_patterns = [
            r'(?:Type|Entity|实体类型)[^\w]*:?\s*([A-Za-z0-9_\u4e00-\u9fa5]+)[^\w]*(?:Value|Text|Value\/Text|值)[^\w]*:?\s*([^\n,;]+)',
            r'([A-Za-z0-9_\u4e00-\u9fa5]+)\s*:\s*([^\n,;]+)',
        ]
        
        for pattern in entity_patterns:
            for match in re.finditer(pattern, response):
                entity_type = match.group(1).strip()
                entity_value = match.group(2).strip()
                
                # 规范化实体类型
                entity_type = self._normalize_entity_type(entity_type)
                
                # 查找位置
                start, end = self._find_text_position(original_text, entity_value)
                if start != -1:
                    entity = {
                        'entity_type': entity_type,
                        'value': entity_value,
                        'start': start,
                        'end': end,
                        'confidence': self.default_score
                    }
                    results.append(entity)
        
        # 如果上面的模式没有找到任何实体，尝试更宽松的方式
        if not results:
            # 查找冒号分隔的键值对
            lines = [line.strip() for line in response.split('\n') if line.strip()]
            for line in lines:
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        entity_type = parts[0].strip()
                        entity_value = parts[1].strip()
                        
                        # 规范化实体类型
                        entity_type = self._normalize_entity_type(entity_type)
                        
                        # 查找位置
                        start, end = self._find_text_position(original_text, entity_value)
                        if start != -1:
                            entity = {
                                'entity_type': entity_type,
                                'value': entity_value,
                                'start': start,
                                'end': end,
                                'confidence': self.default_score
                            }
                            results.append(entity)
        
        return results