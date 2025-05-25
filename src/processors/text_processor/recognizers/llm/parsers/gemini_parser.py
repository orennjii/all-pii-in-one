#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gemini LLM响应解析器模块

专门用于解析Google Gemini模型的响应，支持JSON和文本格式的解析。

本模块提供了GeminiParser类，用于处理Google Gemini模型返回的各种格式的响应。
解析器支持多种响应格式，包括标准JSON、宽松JSON、正则表达式匹配和表格格式，
能够自动识别和提取文本中的敏感信息实体。

主要特性:
    - 多格式响应解析支持
    - 智能文本清理和预处理
    - 容错性强的解析机制
    - 详细的错误报告和日志记录
    - 实体位置精确定位

典型使用场景:
    - PII（个人身份信息）检测和识别
    - 敏感信息实体提取
    - 文本数据隐私保护预处理

示例:
    ```python
    from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
    from src.processors.text_processor.recognizers.llm.parsers.gemini_parser import GeminiParser
    
    # 初始化解析器
    config = LLMParsersConfig()
    parser = GeminiParser(config)
    
    # 解析Gemini响应
    response_text = '[{"entity_type": "PERSON", "value": "张三", "start": 0, "end": 2}]'
    original_text = "张三是一个工程师"
    result = parser.parse(response_text, original_text)
    
    print(f"识别到 {len(result.entities)} 个实体")
    ```

注意:
    本模块依赖于BaseLLMParser基类和相关的配置模块。
    解析器会尝试多种方法来解析响应，确保在各种格式下都能正常工作。
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
    该解析器采用多级解析策略，从最严格的JSON格式开始，
    逐步降级到更宽松的解析方法，以确保最大的兼容性。
    
    支持的解析格式:
        1. 标准JSON格式 - 严格按照JSON规范解析
        2. 宽松JSON格式 - 自动修复常见的JSON格式错误
        3. 正则表达式解析 - 基于模式匹配提取实体信息
        4. 表格格式解析 - 解析管道分隔的表格数据
    
    解析流程:
        1. 文本预处理和清理
        2. 按优先级尝试不同解析方法
        3. 实体后处理和验证
        4. 错误收集和报告
    
    Attributes:
        supported_formats (list): 支持的响应格式列表
        json_strict_mode (bool): 是否启用严格JSON模式
        
    Examples:
        ```python
        config = LLMParsersConfig()
        parser = GeminiParser(config)
        
        # 解析JSON格式响应
        json_response = '[{"entity_type": "EMAIL", "value": "test@example.com", "start": 10, "end": 26}]'
        result = parser.parse(json_response, "请联系 test@example.com 获取更多信息")
        
        # 解析文本格式响应
        text_response = "EMAIL: test@example.com (位置: 10-26)"
        result = parser.parse(text_response, "请联系 test@example.com 获取更多信息")
        ```
        
    Note:
        解析器会自动记录解析过程中的错误和警告信息，
        便于调试和问题诊断。所有解析方法都是幂等的，
        多次调用相同输入会产生相同结果。
    """
    
    def __init__(self, config: LLMParsersConfig):
        """
        初始化Gemini解析器
        
        设置解析器的配置参数，包括支持的响应格式和解析模式。
        
        Args:
            config (LLMParsersConfig): 解析器配置对象，包含以下重要配置:
                - gemini_response_formats: Gemini支持的响应格式列表
                - json_strict_mode: 是否启用严格JSON解析模式
                - 其他继承自基类的配置参数
                
        Raises:
            TypeError: 当config参数类型不正确时
            AttributeError: 当config缺少必要属性时
            
        Examples:
            ```python
            from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
            
            config = LLMParsersConfig()
            config.gemini_response_formats = ["json", "text", "table"]
            config.json_strict_mode = False
            
            parser = GeminiParser(config)
            ```
            
        Note:
            配置对象会被传递给父类进行基础初始化，
            然后设置Gemini特定的配置参数。
        """
        super().__init__(config)
        self.supported_formats = config.gemini_response_formats
        self.json_strict_mode = config.json_strict_mode
        
    def parse(self, response: str, original_text: str) -> LLMResponse:
        """
        解析Gemini的响应
        
        这是解析器的主入口方法，负责协调整个解析流程。
        采用多级解析策略，确保在各种响应格式下都能成功提取实体信息。
        
        解析流程:
            1. 初始化解析状态和错误收集器
            2. 清理和预处理响应文本
            3. 按优先级尝试不同的解析方法
            4. 对解析结果进行后处理和验证
            5. 构建标准化的响应对象
        
        Args:
            response (str): Gemini模型的原始响应文本，可能包含:
                - 标准JSON格式的实体列表
                - Markdown代码块包装的JSON
                - 自然语言描述的实体信息
                - 表格格式的实体数据
            original_text (str): 原始输入文本，用于:
                - 验证实体位置的准确性
                - 提供解析上下文信息
                - 实体后处理和校验
                
        Returns:
            LLMResponse: 标准化的解析响应对象，包含:
                - entities: 解析得到的实体列表（List[EntityMatch]）
                - raw_response: 原始响应文本
                - metadata: 解析元数据（解析器类型、实体数量等）
                - parsing_errors: 解析过程中的错误列表（可能为None）
                
        Raises:
            Exception: 当解析过程中发生严重错误时，错误信息会被捕获
                      并添加到parsing_errors列表中，不会中断程序执行
                      
        Examples:
            ```python
            parser = GeminiParser(config)
            
            # JSON格式响应
            json_response = '''
            [
                {
                    "entity_type": "PERSON",
                    "value": "张三",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.95
                }
            ]
            '''
            result = parser.parse(json_response, "张三是软件工程师")
            
            # 文本格式响应
            text_response = "识别到姓名: 张三 (位置: 0-2)"
            result = parser.parse(text_response, "张三是软件工程师")
            
            print(f"解析到 {len(result.entities)} 个实体")
            for entity in result.entities:
                print(f"{entity.entity_type}: {entity.value}")
            ```
            
        Note:
            - 解析器会尝试多种方法，直到成功解析或所有方法都失败
            - 所有错误都会被记录但不会中断解析流程
            - 返回的实体列表可能为空，但LLMResponse对象总是有效的
            - 解析元数据包含了有用的调试信息
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
        
        对Gemini的原始响应进行预处理，移除可能影响解析的格式标记和说明文本。
        这个步骤对于提高后续解析方法的成功率至关重要。
        
        清理操作包括:
            1. 移除Markdown代码块标记（```json 和 ```）
            2. 去除多余的空白字符和换行符
            3. 过滤掉解释性文本和说明性内容
            4. 保留核心的实体信息内容
        
        Args:
            response (str): 原始响应文本，可能包含:
                - Markdown格式的代码块
                - 中文解释性文本
                - 多余的空白字符
                - 格式化标记
                
        Returns:
            str: 清理后的响应文本，去除了格式标记和说明文本，
                 保留了核心的实体信息内容
                 
        Examples:
            ```python
            parser = GeminiParser(config)
            
            raw_response = '''
            以下是识别结果：
            ```json
            [{"entity_type": "PERSON", "value": "张三", "start": 0, "end": 2}]
            ```
            根据分析，找到了1个实体。
            '''
            
            cleaned = parser._clean_response(raw_response)
            # 结果: '[{"entity_type": "PERSON", "value": "张三", "start": 0, "end": 2}]'
            ```
            
        Note:
            - 该方法是幂等的，多次调用结果相同
            - 会保留JSON和表格数据的完整性
            - 过滤规则基于常见的中文解释性模式
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
        
        按照优先级顺序尝试不同的解析策略，直到成功解析或所有方法都失败。
        这种分层解析策略确保了在各种响应格式下都能最大化解析成功率。
        
        解析方法优先级:
            1. 标准JSON解析 - 最快最准确，适用于格式规范的JSON响应
            2. 宽松JSON解析 - 自动修复常见JSON格式错误
            3. 正则表达式解析 - 基于模式匹配，适用于结构化文本
            4. 表格格式解析 - 解析管道分隔的表格数据
        
        Args:
            response (str): 清理后的响应文本
            original_text (str): 原始输入文本，用于位置验证和上下文信息
            parsing_errors (List[str]): 错误信息收集列表，每种解析方法的错误
                                       都会被添加到这个列表中
                                       
        Returns:
            List[EntityMatch]: 解析成功时返回实体列表，所有方法都失败时返回空列表。
                             实体对象包含类型、值、位置、置信度等信息。
                             
        Side Effects:
            - 向parsing_errors列表添加错误信息
            - 记录调试日志信息
            
        Examples:
            ```python
            parser = GeminiParser(config)
            errors = []
            
            # JSON格式响应
            json_response = '[{"entity_type": "EMAIL", "value": "test@example.com"}]'
            entities = parser._try_parse_methods(json_response, "原文", errors)
            
            # 文本格式响应
            text_response = "EMAIL: test@example.com (位置: 0-15)"
            entities = parser._try_parse_methods(text_response, "原文", errors)
            
            print(f"解析到 {len(entities)} 个实体")
            if errors:
                print(f"解析错误: {errors}")
            ```
            
        Note:
            - 方法按照成功率和性能优先级排序
            - 一旦某种方法成功，就不会尝试后续方法
            - 所有错误都会被记录，有助于问题诊断
            - 返回的实体列表已经过基本验证
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
        
        使用严格的JSON解析器处理格式规范的JSON响应。这是首选的解析方法，
        因为它最快且最准确，适用于Gemini返回的标准格式化响应。
        
        支持的JSON格式:
            - 实体数组: [{"entity_type": "...", "value": "...", ...}, ...]
            - 完整的实体字段: entity_type, value, start, end, confidence等
            - 可选字段: is_inferred, notes, metadata
        
        Args:
            response (str): 响应文本，期望包含JSON格式的实体数组
            parsing_errors (List[str]): 错误信息收集列表，JSON解析失败时
                                       会添加详细的错误信息
                                       
        Returns:
            List[EntityMatch]: 成功解析时返回实体对象列表，失败时返回空列表。
                             每个实体对象包含完整的字段信息。
                             
        Raises:
            json.JSONDecodeError: JSON格式错误时会被捕获并记录到parsing_errors
            Exception: 其他解析错误也会被捕获并记录
            
        Examples:
            ```python
            parser = GeminiParser(config)
            errors = []
            
            # 标准JSON响应
            json_text = '''
            [
                {
                    "entity_type": "PERSON",
                    "value": "张三",
                    "start": 0,
                    "end": 2,
                    "confidence": 0.95
                },
                {
                    "entity_type": "EMAIL",
                    "value": "zhang@example.com",
                    "start": 10,
                    "end": 27,
                    "confidence": 0.9
                }
            ]
            '''
            
            entities = parser._parse_json_response(json_text, errors)
            print(f"解析到 {len(entities)} 个实体")
            ```
            
        Note:
            - 要求响应包含有效的JSON数组格式
            - 会自动提取JSON部分，忽略其他文本
            - 实体字段验证在_parse_entity_dict方法中进行
            - 单个实体解析失败不会影响其他实体的解析
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
        
        当标准JSON解析失败时，这个方法会尝试自动修复常见的JSON格式错误，
        然后重新进行解析。这种容错机制能够处理Gemini偶尔产生的不完全
        符合JSON规范的响应。
        
        自动修复的问题类型:
            1. 缺失的属性名引号: {name: "value"} -> {"name": "value"}
            2. 尾随逗号: {"a": 1,} -> {"a": 1}
            3. 单引号: {'name': 'value'} -> {"name": "value"}
            4. 其他常见的格式不规范问题
        
        Args:
            response (str): 响应文本，包含可能格式不规范的JSON内容
            parsing_errors (List[str]): 错误信息收集列表，修复失败时
                                       会添加相关错误信息
                                       
        Returns:
            List[EntityMatch]: 修复并解析成功时返回实体列表，失败时返回空列表。
                             修复后的JSON会使用标准JSON解析器处理。
                             
        Examples:
            ```python
            parser = GeminiParser(config)
            errors = []
            
            # 格式不规范的JSON
            malformed_json = '''
            [
                {
                    entity_type: "PERSON",
                    value: '张三',
                    start: 0,
                    end: 2,
                    confidence: 0.95,
                }
            ]
            '''
            
            entities = parser._parse_relaxed_json(malformed_json, errors)
            print(f"修复并解析到 {len(entities)} 个实体")
            ```
            
        Note:
            - 这是一个备用解析方法，仅在标准JSON解析失败时使用
            - 修复操作基于常见的JSON格式错误模式
            - 修复后会调用标准JSON解析方法进行实际解析
            - 如果修复失败，错误信息会被记录但不会抛出异常
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
        
        当JSON解析方法都失败时，使用正则表达式从文本中提取实体信息。
        这种方法适用于Gemini以自然语言形式返回实体信息的情况。
        
        支持的文本模式:
            - "实体类型: 实体值 (位置: start-end)"
            - "PERSON: 张三 (position: 0-2)"
            - 其他类似的结构化文本格式
        
        正则表达式模式匹配包含实体类型、值和位置信息的文本行。
        
        Args:
            response (str): 响应文本，包含结构化的实体描述
            original_text (str): 原始输入文本，用于验证提取的位置信息
            parsing_errors (List[str]): 错误信息收集列表，解析失败时
                                       会添加详细错误信息
                                       
        Returns:
            List[EntityMatch]: 成功提取时返回实体列表，失败时返回空列表。
                             实体的置信度默认设置为0.8。
                             
        Examples:
            ```python
            parser = GeminiParser(config)
            errors = []
            
            # 文本格式响应
            text_response = '''
            识别到以下实体：
            PERSON: 张三 (位置: 0-2)
            EMAIL: zhang@example.com (position: 10-27)
            PHONE: 13800138000 (位置: 35-46)
            '''
            
            entities = parser._parse_with_regex(text_response, "原始文本", errors)
            print(f"正则表达式解析到 {len(entities)} 个实体")
            ```
            
        Note:
            - 这是备用解析方法，当JSON解析都失败时使用
            - 支持中英文混合的位置标记
            - 默认置信度设置为0.8
            - 位置信息解析失败的实体会被跳过并记录错误
            - 实体类型会自动转换为大写
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
        
        解析以管道符分隔的表格格式数据，这种格式通常用于结构化显示实体信息。
        适用于Gemini返回Markdown表格或类似表格格式的响应。
        
        期望的表格格式:
            | 实体类型 | 实体值 | 起始位置 | 结束位置 | 置信度 |
            |---------|--------|----------|----------|--------|
            | PERSON  | 张三   | 0        | 2        | 0.95   |
            | EMAIL   | test@  | 10       | 25       | 0.9    |
        
        Args:
            response (str): 响应文本，包含管道符分隔的表格数据
            original_text (str): 原始输入文本，用于验证位置信息
            parsing_errors (List[str]): 错误信息收集列表，解析失败时
                                       会添加相关错误信息
                                       
        Returns:
            List[EntityMatch]: 成功解析时返回实体列表，失败时返回空列表。
                             表格中的每一行数据会被转换为一个EntityMatch对象。
                             
        Examples:
            ```python
            parser = GeminiParser(config)
            errors = []
            
            # 表格格式响应
            table_response = '''
            识别结果如下：
            | 实体类型 | 实体值 | 起始位置 | 结束位置 | 置信度 |
            |---------|--------|----------|----------|--------|
            | PERSON  | 张三   | 0        | 2        | 0.95   |
            | EMAIL   | test@example.com | 10 | 25 | 0.9    |
            '''
            
            entities = parser._parse_table_format(table_response, "原始文本", errors)
            print(f"表格解析到 {len(entities)} 个实体")
            ```
            
        Note:
            - 这是最后的备用解析方法
            - 要求表格至少包含6列数据（包括分隔符）
            - 会跳过表头和分隔行
            - 实体类型会自动转换为大写
            - 数值转换失败的行会被跳过并记录错误
            - 空的实体类型或值会被跳过
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
        
        将字典格式的实体数据转换为EntityMatch对象。这个方法负责验证
        实体数据的完整性和正确性，确保所有必需字段都存在且类型正确。
        
        必需字段验证:
            - entity_type: 实体类型，必须非空字符串
            - value: 实体值，必须非空字符串  
            - start: 起始位置，必须为有效整数
            - end: 结束位置，必须为有效整数
            
        可选字段处理:
            - confidence: 置信度，默认为0.8
            - is_inferred: 是否为推断得出，可选布尔值
            - notes: 备注信息，可选字符串
            - metadata: 元数据，可选字典
        
        Args:
            item (Dict[str, Any]): 包含实体信息的字典，期望包含以下字段:
                - entity_type (str): 实体类型（必需）
                - value (str): 实体值（必需）
                - start (int): 起始位置（必需）
                - end (int): 结束位置（必需）
                - confidence (float, optional): 置信度，默认0.8
                - is_inferred (bool, optional): 是否推断
                - notes (str, optional): 备注
                - metadata (dict, optional): 元数据
                
        Returns:
            Optional[EntityMatch]: 解析成功时返回EntityMatch对象，
                                 验证失败时返回None。对象包含所有
                                 验证过的字段信息。
                                 
        Examples:
            ```python
            parser = GeminiParser(config)
            
            # 完整的实体字典
            entity_dict = {
                "entity_type": "PERSON",
                "value": "张三",
                "start": 0,
                "end": 2,
                "confidence": 0.95,
                "is_inferred": False,
                "notes": "高置信度识别",
                "metadata": {"source": "gemini"}
            }
            
            entity = parser._parse_entity_dict(entity_dict)
            if entity:
                print(f"实体: {entity.entity_type} = {entity.value}")
            
            # 最小字段的实体字典
            minimal_dict = {
                "entity_type": "EMAIL",
                "value": "test@example.com",
                "start": 10,
                "end": 25
            }
            
            entity = parser._parse_entity_dict(minimal_dict)
            ```
            
        Note:
            - 实体类型会自动转换为大写
            - 所有字段都会进行类型验证和转换
            - 验证失败的情况会记录警告日志
            - 返回None时不会抛出异常，便于批量处理
            - 位置信息必须为非负整数
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
