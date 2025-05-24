#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LLM响应解析器基类模块

定义所有LLM响应解析器必须实现的接口和通用功能。
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from src.commons.loggers import get_module_logger
from src.configs.processors.text_processor.recognizers.llm import LLMParsersConfig
from src.processors.text_processor.recognizers.llm.parsers.entity_match import EntityMatch

logger = get_module_logger(__name__)


@dataclass
class LLMResponse:
    """
    LLM响应的标准化数据结构
    
    用于封装解析后的LLM响应，包括识别到的实体列表和元数据。
    """
    
    entities: List[EntityMatch]
    """识别到的实体列表"""
    
    raw_response: str
    """原始LLM响应"""
    
    metadata: Optional[Dict[str, Any]] = None
    """响应的元数据信息"""
    
    parsing_errors: Optional[List[str]] = None
    """解析过程中的错误信息"""


class BaseLLMParser(ABC):
    """
    LLM响应解析器基类
    
    所有LLM响应解析器必须继承此类并实现其抽象方法。
    提供了通用的验证和处理功能。
    """
    
    def __init__(self, config: LLMParsersConfig):
        """
        初始化解析器
        
        Args:
            config: 解析器配置
        """
        self.config = config
        self.min_confidence = config.min_confidence
        
    @abstractmethod
    def parse(self, response: str, original_text: str) -> LLMResponse:
        """
        解析LLM响应
        
        Args:
            response: LLM的原始响应文本
            original_text: 原始输入文本
            
        Returns:
            LLMResponse: 解析后的标准化响应
            
        Raises:
            ValueError: 当响应格式无效时
            Exception: 当解析过程出错时
        """
        pass
    
    def validate_entity_positions(
        self, 
        entities: List[EntityMatch], 
        original_text: str
    ) -> List[EntityMatch]:
        """
        验证实体位置的有效性
        
        Args:
            entities: 待验证的实体列表
            original_text: 原始文本
            
        Returns:
            List[EntityMatch]: 验证通过的实体列表
        """
        validated_entities = []
        
        for entity in entities:
            try:
                # 检查位置边界
                if (entity.start < 0 or 
                    entity.end > len(original_text) or 
                    entity.start >= entity.end):
                    logger.warning(
                        f"实体位置无效: {entity.entity_type} "
                        f"({entity.start}, {entity.end}), 文本长度: {len(original_text)}"
                    )
                    continue
                
                # 验证实体值是否与位置匹配
                extracted_value = original_text[entity.start:entity.end]
                if extracted_value != entity.value:
                    logger.warning(
                        f"实体值不匹配: 期望 '{entity.value}', "
                        f"实际 '{extracted_value}' 在位置 ({entity.start}, {entity.end})"
                    )
                    # 尝试在附近查找正确位置
                    corrected_entity = self._try_correct_position(
                        entity, original_text
                    )
                    if corrected_entity:
                        validated_entities.append(corrected_entity)
                    continue
                
                # 检查置信度阈值
                if entity.confidence < self.min_confidence:
                    logger.debug(
                        f"实体置信度过低: {entity.entity_type} "
                        f"confidence={entity.confidence:.2f} < {self.min_confidence}"
                    )
                    continue
                
                validated_entities.append(entity)
                
            except Exception as e:
                logger.warning(f"验证实体时出错: {str(e)}")
                continue
        
        return validated_entities
    
    def _try_correct_position(
        self, 
        entity: EntityMatch, 
        original_text: str
    ) -> Optional[EntityMatch]:
        """
        尝试修正实体的位置信息
        
        Args:
            entity: 原始实体
            original_text: 原始文本
            
        Returns:
            Optional[EntityMatch]: 修正后的实体，如果无法修正则返回None
        """
        search_value = entity.value.strip()
        
        # 在原文中查找匹配的子字符串
        start_pos = original_text.find(search_value)
        if start_pos != -1:
            end_pos = start_pos + len(search_value)
            logger.debug(
                f"修正实体位置: {entity.entity_type} "
                f"从 ({entity.start}, {entity.end}) 到 ({start_pos}, {end_pos})"
            )
            
            return EntityMatch(
                entity_type=entity.entity_type,
                value=search_value,
                start=start_pos,
                end=end_pos,
                confidence=entity.confidence,
                is_inferred=entity.is_inferred,
                notes=entity.notes,
                metadata=entity.metadata
            )
        
        return None
    
    def filter_overlapping_entities(
        self, 
        entities: List[EntityMatch]
    ) -> List[EntityMatch]:
        """
        过滤重叠的实体，保留置信度更高的实体
        
        Args:
            entities: 实体列表
            
        Returns:
            List[EntityMatch]: 过滤后的实体列表
        """
        if not entities:
            return []
        
        # 按起始位置排序
        sorted_entities = sorted(entities, key=lambda x: (x.start, -x.confidence))
        filtered_entities = []
        
        for entity in sorted_entities:
            # 检查是否与已选择的实体重叠
            is_overlapping = False
            for selected_entity in filtered_entities:
                if (entity.start < selected_entity.end and 
                    entity.end > selected_entity.start):
                    # 有重叠，比较置信度
                    if entity.confidence > selected_entity.confidence:
                        # 移除置信度较低的实体
                        filtered_entities.remove(selected_entity)
                        logger.debug(
                            f"替换重叠实体: {selected_entity.entity_type} "
                            f"(confidence={selected_entity.confidence:.2f}) "
                            f"-> {entity.entity_type} "
                            f"(confidence={entity.confidence:.2f})"
                        )
                        break
                    else:
                        # 当前实体置信度较低，跳过
                        is_overlapping = True
                        break
            
            if not is_overlapping:
                filtered_entities.append(entity)
        
        return filtered_entities
    
    def post_process_entities(
        self, 
        entities: List[EntityMatch], 
        original_text: str
    ) -> List[EntityMatch]:
        """
        对实体进行后处理
        
        Args:
            entities: 原始实体列表
            original_text: 原始文本
            
        Returns:
            List[EntityMatch]: 后处理后的实体列表
        """
        # 验证位置
        validated_entities = self.validate_entity_positions(entities, original_text)
        
        # 过滤重叠实体
        filtered_entities = self.filter_overlapping_entities(validated_entities)
        
        # 按位置排序
        sorted_entities = sorted(filtered_entities, key=lambda x: x.start)
        
        logger.debug(
            f"后处理完成: 原始 {len(entities)} 个实体 "
            f"-> 最终 {len(sorted_entities)} 个实体"
        )
        
        return sorted_entities
