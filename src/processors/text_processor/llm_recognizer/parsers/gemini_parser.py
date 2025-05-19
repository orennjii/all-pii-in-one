import json
import logging
import re
from typing import List, Dict, Any, Optional, Union

from .base_parser import BaseParser

logger = logging.getLogger("llm_recognizer.gemini_parser")


class GeminiParser(BaseParser):
    """
    针对Google Gemini模型输出的专用解析器。
    
    此解析器能够处理Gemini模型的各种输出格式，包括:
    1. 结构化JSON响应
    2. 标记了实体的文本
    3. 基于特定提示模板的自由文本响应
    """
    
    def __init__(
        self, 
        expected_format: str = "json", 
        fallback_to_text: bool = True,
        default_score: float = 0.8,
        **kwargs
    ):
        """
        初始化Gemini解析器。
        
        Args:
            expected_format: 期望的输出格式，支持 "json" 或 "text"
            fallback_to_text: 解析JSON失败时是否回退到文本解析
            default_score: 当LLM未提供置信度分数时使用的默认值
            **kwargs: 传递给基类的额外参数
        """
        super().__init__(**kwargs)
        self.expected_format = expected_format.lower()
        self.fallback_to_text = fallback_to_text
        self.default_score = default_score
        
        logger.debug(f"初始化GeminiParser，期望格式: {expected_format}")
    
    def parse(
        self, 
        llm_response: str, 
        original_text: str, 
        entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析Gemini响应并提取实体信息。
        
        Args:
            llm_response: Gemini返回的响应文本
            original_text: 发送给Gemini的原始文本
            entities: 要查找的实体类型列表（可选）
            
        Returns:
            List[Dict[str, Any]]: 标准化的实体信息列表
        """
        logger.debug(f"开始解析Gemini响应，长度: {len(llm_response)} 字符")
        
        # 根据期望格式选择解析方法
        if self.expected_format == "json":
            try:
                return self._parse_json_response(llm_response, original_text, entities)
            except Exception as e:
                logger.warning(f"JSON解析失败: {str(e)}")
                if self.fallback_to_text:
                    logger.info("回退到文本解析")
                    return self._parse_text_response(llm_response, original_text, entities)
                else:
                    return []
        else:
            # 默认使用文本解析
            return self._parse_text_response(llm_response, original_text, entities)
    
    def _parse_json_response(
        self, 
        response: str, 
        original_text: str, 
        allowed_entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析JSON格式的Gemini响应。
        
        Args:
            response: Gemini的JSON响应文本
            original_text: 原始文本
            allowed_entities: 允许的实体类型列表
            
        Returns:
            List[Dict[str, Any]]: 解析后的实体列表
        """
        # 提取JSON部分
        json_str = self._extract_json(response)
        
        try:
            # 解析JSON
            parsed = json.loads(json_str)
            
            # 处理各种可能的JSON结构
            entities_list = []
            
            # 情况1: 直接是实体列表
            if isinstance(parsed, list):
                entities_list = parsed
            # 情况2: 有一个"entities"或"results"字段
            elif isinstance(parsed, dict):
                if "entities" in parsed:
                    entities_list = parsed["entities"]
                elif "results" in parsed:
                    entities_list = parsed["results"]
                elif "data" in parsed:
                    entities_list = parsed["data"]
            
            # 规范化和验证结果
            results = []
            for entity in entities_list:
                # 确保有实体类型
                if "entity_type" not in entity and "type" in entity:
                    entity["entity_type"] = entity["type"]
                    
                # 规范化实体类型
                if "entity_type" in entity:
                    entity["entity_type"] = self._normalize_entity_type(entity["entity_type"])
                
                # 如果有文本但没有位置信息，尝试查找位置
                if ("text" in entity or "value" in entity) and not ("start" in entity and "end" in entity):
                    entity_text = entity.get("text") or entity.get("value")
                    start, end = self._find_text_position(original_text, entity_text)
                    if start >= 0:
                        entity["start"] = start
                        entity["end"] = end
                
                # 添加默认分数（如果没有）
                if "score" not in entity:
                    entity["score"] = self.default_score
                
                # 验证并添加到结果
                if self._validate_entity(entity, allowed_entities):
                    results.append(entity)
                else:
                    logger.debug(f"忽略无效实体: {entity}")
            
            return results
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析错误: {str(e)}")
            raise
    
    def _parse_text_response(
        self, 
        response: str, 
        original_text: str, 
        allowed_entities: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        解析自由文本格式的Gemini响应。
        
        Args:
            response: Gemini的文本响应
            original_text: 原始文本
            allowed_entities: 允许的实体类型列表
            
        Returns:
            List[Dict[str, Any]]: 解析后的实体列表
        """
        results = []
        
        # 尝试使用正则表达式提取格式化的实体
        # 寻找类似 "PERSON: John Doe (位置: 10-18)" 或 "EMAIL: user@example.com"
        entity_pattern = r"([A-Z_]+):\s*([^\(\n]+)(?:\s*\(位置:\s*(\d+)-(\d+)\))?"
        entity_matches = re.finditer(entity_pattern, response)
        
        for match in entity_matches:
            entity_type = match.group(1).strip()
            entity_text = match.group(2).strip()
            
            # 尝试获取位置信息
            if match.group(3) and match.group(4):
                start = int(match.group(3))
                end = int(match.group(4))
            else:
                # 如果没有提供位置，尝试在原文中查找
                start, end = self._find_text_position(original_text, entity_text)
            
            # 如果找到了位置
            if start >= 0:
                entity = {
                    "entity_type": self._normalize_entity_type(entity_type),
                    "start": start,
                    "end": end,
                    "value": entity_text,
                    "score": self.default_score
                }
                
                # 验证并添加到结果
                if self._validate_entity(entity, allowed_entities):
                    results.append(entity)
        
        # 如果上面的方法没有找到任何实体，尝试使用备用方法
        if not results:
            # 查找可能的实体类型提及
            if allowed_entities:
                for entity_type in allowed_entities:
                    # 查找类似 "找到 PERSON: John Doe" 的模式
                    pattern = rf"{entity_type}(?::|：)\s*([^\n,\.]+)"
                    matches = re.finditer(pattern, response, re.IGNORECASE)
                    
                    for match in matches:
                        entity_text = match.group(1).strip()
                        start, end = self._find_text_position(original_text, entity_text)
                        
                        if start >= 0:
                            entity = {
                                "entity_type": entity_type,
                                "start": start,
                                "end": end,
                                "value": entity_text,
                                "score": self.default_score * 0.8  # 降低置信度，因为这是备用方法
                            }
                            results.append(entity)
        
        return results
    
    def _extract_json(self, text: str) -> str:
        """
        从文本中提取JSON部分。
        
        Args:
            text: 可能包含JSON的文本
            
        Returns:
            str: 提取的JSON字符串
        """
        # 尝试查找代码块中的JSON (```json ... ```)
        json_block_match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text)
        if json_block_match:
            return json_block_match.group(1).strip()
        
        # 尝试查找第一个左大括号到最后一个右大括号之间的内容
        # 这种方法会提取到最外层的JSON对象
        first_brace = text.find('{')
        last_brace = text.rfind('}')
        if first_brace >= 0 and last_brace > first_brace:
            return text[first_brace:last_brace+1].strip()
        
        # 尝试查找方括号格式的JSON数组
        first_bracket = text.find('[')
        last_bracket = text.rfind(']')
        if first_bracket >= 0 and last_bracket > first_bracket:
            return text[first_bracket:last_bracket+1].strip()
        
        # 如果找不到明确的JSON结构，返回整个文本
        # 这可能会导致JSON解析错误，但我们会在调用方法中处理这种情况
        return text