#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini LLM响应解析器模块

专门用于解析Google Gemini模型的响应，支持JSON和文本格式的解析。
"""

import json
import re
from typing import List, Dict, Any, Optional

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
from src.processors.text_processor.recognizers.llm.parsers.base_parser import (
    BaseLLMParser, LLMResponse
)
from src.processors.text_processor.recognizers.llm.parsers.entity_match import EntityMatch

logger = get_module_logger(__name__)


class GeminiParser(BaseLLMParser):
    """
    Google Gemini响应解析器
    
    专门处理Gemini模型的响应格式，支持多种响应格式的解析。
    """
    
    def __init__(self, config: LLMParsersConfig):
        """
        初始化Gemini解析器
        
        Args:
            config: 解析器配置
        """
        super().__init__(config)
        self.supported_formats = config.gemini_response_formats
        self.json_strict_mode = config.json_strict_mode
        
    def parse(self, response: str, original_text: str) -> LLMResponse:
        """
        解析Gemini的响应
        
        Args:
            response: Gemini的原始响应
            original_text: 原始输入文本
            
        Returns:
            LLMResponse: 解析后的标准化响应
        """
        entities = []
        parsing_errors = []
        metadata = {"parser_type": "gemini"}
        
        try:
            # 清理响应文本
            cleaned_response = self._clean_response(response)
            
            # 尝试不同的解析方法
            entities = self._try_parse_methods(
                cleaned_response, original_text, parsing_errors
            )
            
            # 后处理实体
            entities = self.post_process_entities(entities, original_text)
            
            metadata["parsed_entities_count"] = str(len(entities))
            
        except Exception as e:
            error_msg = f"解析Gemini响应时出错: {str(e)}"
            logger.error(error_msg)
            parsing_errors.append(error_msg)
        
        return LLMResponse(
            entities=entities,
            raw_response=response,
            metadata=metadata,
            parsing_errors=parsing_errors if parsing_errors else None
        )
    
    def _clean_response(self, response: str) -> str:
        """
        清理响应文本，移除不必要的内容
        
        Args:
            response: 原始响应
            
        Returns:
            str: 清理后的响应
        """
        # 移除Markdown代码块标记
        response = re.sub(r'```json\s*', '', response)
        response = re.sub(r'```\s*$', '', response)
        
        # 移除多余的空白字符
        response = response.strip()
        
        # 移除可能的说明文本
        lines = response.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # 跳过解释性文本
            if (line.startswith('以下是') or 
                line.startswith('根据') or 
                line.startswith('识别结果') or
                '如下' in line):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines).strip()
    
    def _try_parse_methods(
        self, 
        response: str, 
        original_text: str, 
        parsing_errors: List[str]
    ) -> List[EntityMatch]:
        """
        尝试多种解析方法
        
        Args:
            response: 清理后的响应
            original_text: 原始文本
            parsing_errors: 错误列表
            
        Returns:
            List[EntityMatch]: 解析得到的实体列表
        """
        # 方法1: 标准JSON解析
        entities = self._parse_json_response(response, parsing_errors)
        if entities:
            logger.debug(f"JSON解析成功，获得 {len(entities)} 个实体")
            return entities
        
        # 方法2: 宽松JSON解析
        entities = self._parse_relaxed_json(response, parsing_errors)
        if entities:
            logger.debug(f"宽松JSON解析成功，获得 {len(entities)} 个实体")
            return entities
        
        # 方法3: 正则表达式解析
        entities = self._parse_with_regex(response, original_text, parsing_errors)
        if entities:
            logger.debug(f"正则表达式解析成功，获得 {len(entities)} 个实体")
            return entities
        
        # 方法4: 表格格式解析
        entities = self._parse_table_format(response, original_text, parsing_errors)
        if entities:
            logger.debug(f"表格格式解析成功，获得 {len(entities)} 个实体")
            return entities
        
        logger.warning("所有解析方法都失败了")
        return []
    
    def _parse_json_response(
        self, 
        response: str, 
        parsing_errors: List[str]
    ) -> List[EntityMatch]:
        """
        解析标准JSON格式的响应
        
        Args:
            response: 响应文本
            parsing_errors: 错误列表
            
        Returns:
            List[EntityMatch]: 解析得到的实体列表
        """
        try:
            # 提取JSON部分
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if not json_match:
                return []
            
            json_str = json_match.group(0)
            data = json.loads(json_str)
            
            if not isinstance(data, list):
                parsing_errors.append("JSON响应不是数组格式")
                return []
            
            entities = []
            for item in data:
                try:
                    entity = self._parse_entity_dict(item)
                    if entity:
                        entities.append(entity)
                except Exception as e:
                    parsing_errors.append(f"解析实体项失败: {str(e)}")
                    continue
            
            return entities
            
        except json.JSONDecodeError as e:
            parsing_errors.append(f"JSON解析失败: {str(e)}")
            return []
        except Exception as e:
            parsing_errors.append(f"标准JSON解析出错: {str(e)}")
            return []
    
    def _parse_relaxed_json(
        self, 
        response: str, 
        parsing_errors: List[str]
    ) -> List[EntityMatch]:
        """
        宽松的JSON解析，尝试修复常见的JSON格式问题
        
        Args:
            response: 响应文本
            parsing_errors: 错误列表
            
        Returns:
            List[EntityMatch]: 解析得到的实体列表
        """
        try:
            # 修复常见的JSON格式问题
            fixed_response = response
            
            # 修复缺失的引号
            fixed_response = re.sub(r'(\w+):', r'"\1":', fixed_response)
            
            # 修复尾随逗号
            fixed_response = re.sub(r',\s*}', '}', fixed_response)
            fixed_response = re.sub(r',\s*]', ']', fixed_response)
            
            # 修复单引号
            fixed_response = fixed_response.replace("'", '"')
            
            return self._parse_json_response(fixed_response, parsing_errors)
            
        except Exception as e:
            parsing_errors.append(f"宽松JSON解析出错: {str(e)}")
            return []
    
    def _parse_with_regex(
        self, 
        response: str, 
        original_text: str, 
        parsing_errors: List[str]
    ) -> List[EntityMatch]:
        """
        使用正则表达式解析响应
        
        Args:
            response: 响应文本
            original_text: 原始文本
            parsing_errors: 错误列表
            
        Returns:
            List[EntityMatch]: 解析得到的实体列表
        """
        try:
            entities = []
            
            # 匹配实体模式: 实体类型: 实体值 (位置: start-end)
            pattern = r'(\w+):\s*([^(]+)\s*\((?:位置:|position:)?\s*(\d+)-(\d+)\)'
            matches = re.findall(pattern, response, re.IGNORECASE)
            
            for match in matches:
                entity_type, value, start_str, end_str = match
                try:
                    start = int(start_str)
                    end = int(end_str)
                    
                    entity = EntityMatch(
                        entity_type=entity_type.strip().upper(),
                        value=value.strip(),
                        start=start,
                        end=end,
                        confidence=0.8  # 默认置信度
                    )
                    entities.append(entity)
                    
                except ValueError as e:
                    parsing_errors.append(f"解析位置信息失败: {str(e)}")
                    continue
            
            return entities
            
        except Exception as e:
            parsing_errors.append(f"正则表达式解析出错: {str(e)}")
            return []
    
    def _parse_table_format(
        self, 
        response: str, 
        original_text: str, 
        parsing_errors: List[str]
    ) -> List[EntityMatch]:
        """
        解析表格格式的响应
        
        Args:
            response: 响应文本
            original_text: 原始文本
            parsing_errors: 错误列表
            
        Returns:
            List[EntityMatch]: 解析得到的实体列表
        """
        try:
            entities = []
            lines = response.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line or '|' not in line:
                    continue
                
                # 解析表格行: | 实体类型 | 实体值 | 起始位置 | 结束位置 | 置信度 |
                parts = [part.strip() for part in line.split('|')]
                if len(parts) >= 6:
                    try:
                        entity_type = parts[1]
                        value = parts[2]
                        start = int(parts[3])
                        end = int(parts[4])
                        confidence = float(parts[5])
                        
                        if not entity_type or not value:
                            parsing_errors.append("实体类型或值为空")
                            continue
                        
                        entity = EntityMatch(
                            entity_type=entity_type.upper(),
                            value=value,
                            start=start,
                            end=end,
                            confidence=confidence
                        )
                        entities.append(entity)
                        
                    except (ValueError, IndexError) as e:
                        parsing_errors.append(f"解析表格行失败: {str(e)}")
                        continue
            
            return entities
            
        except Exception as e:
            parsing_errors.append(f"表格格式解析出错: {str(e)}")
            return []
    
    def _parse_entity_dict(self, item: Dict[str, Any]) -> Optional[EntityMatch]:
        """
        从字典解析单个实体
        
        Args:
            item: 实体字典
            
        Returns:
            Optional[EntityMatch]: 解析得到的实体，失败时返回None
        """
        try:
            # 必需字段
            entity_type = item.get("entity_type")
            value = item.get("value")
            start = item.get("start")
            end = item.get("end")
            confidence = item.get("confidence", 0.8)
            
            # 必需字段验证和类型转换
            if not entity_type or not value or start is None or end is None:
                logger.warning(f"实体字典缺少必需字段: {item}")
                return None
            
            try:
                start_int = int(start)
                end_int = int(end)
                confidence_float = float(confidence)
            except (ValueError, TypeError) as e:
                logger.warning(f"类型转换失败: {str(e)}, 数据: {item}")
                return None
            
            # 可选字段
            is_inferred = item.get("is_inferred")
            notes = item.get("notes")
            metadata = item.get("metadata")
            
            return EntityMatch(
                entity_type=str(entity_type).upper(),
                value=str(value),
                start=start_int,
                end=end_int,
                confidence=confidence_float,
                is_inferred=is_inferred,
                notes=notes,
                metadata=metadata
            )
            
        except (ValueError, TypeError) as e:
            logger.warning(f"解析实体字典时出错: {str(e)}, 数据: {item}")
            return None
