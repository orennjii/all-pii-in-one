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
        # 位置容错参数
        self.max_position_offset = getattr(config, 'max_position_offset', 5)  # 最大位置偏移
        self.allow_partial_match = getattr(config, 'allow_partial_match', True)  # 是否允许部分匹配
        
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
                    entity.end = min(entity.end, len(original_text))
                
                # 验证实体值是否与位置匹配
                extracted_value = original_text[entity.start:entity.end]
                if extracted_value != entity.value:
                    # 尝试容错匹配
                    corrected_entity = self._handle_position_mismatch(
                        entity, extracted_value, original_text
                    )
                    if corrected_entity:
                        validated_entities.append(corrected_entity)
                    else:
                        logger.warning(
                            f"实体值不匹配且无法修正: 期望 '{entity.value}', "
                            f"实际 '{extracted_value}' 在位置 ({entity.start}, {entity.end})"
                        )
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
    
    def _handle_position_mismatch(
        self, 
        entity: EntityMatch, 
        extracted_value: str,
        original_text: str
    ) -> Optional[EntityMatch]:
        """
        处理位置不匹配的情况，尝试多种修正策略
        
        Args:
            entity: 原始实体
            extracted_value: 从原始位置提取的实际值
            original_text: 原始文本
            
        Returns:
            Optional[EntityMatch]: 修正后的实体，如果无法修正则返回None
        """
        entity_value = entity.value.strip()
        extracted_clean = extracted_value.strip()
        
        # 策略1：检查是否是部分包含关系
        if self.allow_partial_match:
            # 情况1：提取的值包含实体值（可能有额外字符）
            if entity_value in extracted_clean:
                # 在提取值中找到实体值的精确位置
                relative_start = extracted_clean.find(entity_value)
                absolute_start = entity.start + relative_start
                absolute_end = absolute_start + len(entity_value)
                
                logger.debug(
                    f"部分匹配成功（包含）: {entity.entity_type} "
                    f"位置修正从 ({entity.start}, {entity.end}) "
                    f"到 ({absolute_start}, {absolute_end})"
                )
                
                return EntityMatch(
                    entity_type=entity.entity_type,
                    value=entity_value,
                    start=absolute_start,
                    end=absolute_end,
                    confidence=entity.confidence * 0.9,  # 略微降低置信度
                    is_inferred=entity.is_inferred,
                    notes=entity.notes,
                    metadata=entity.metadata
                )
            
            # 情况2：实体值包含提取的值（可能实体值过长）
            if extracted_clean in entity_value:
                logger.debug(
                    f"部分匹配成功（被包含）: {entity.entity_type} "
                    f"使用提取的值 '{extracted_clean}'"
                )
                
                return EntityMatch(
                    entity_type=entity.entity_type,
                    value=extracted_clean,
                    start=entity.start,
                    end=entity.end,
                    confidence=entity.confidence * 0.8,  # 更多降低置信度
                    is_inferred=entity.is_inferred,
                    notes=entity.notes,
                    metadata=entity.metadata
                )
        
        # 策略2：尝试在附近搜索精确匹配
        corrected_entity = self._try_correct_position(entity, original_text)
        if corrected_entity:
            return corrected_entity
        
        # 策略3：模糊搜索（去除标点符号等）
        fuzzy_entity = self._fuzzy_find_entity(entity, original_text)
        if fuzzy_entity:
            return fuzzy_entity
        
        return None
    
    def _fuzzy_find_entity(
        self, 
        entity: EntityMatch, 
        original_text: str
    ) -> Optional[EntityMatch]:
        """
        模糊搜索实体位置，处理标点符号和空格差异
        
        Args:
            entity: 原始实体
            original_text: 原始文本
            
        Returns:
            Optional[EntityMatch]: 修正后的实体，如果无法找到则返回None
        """
        import re
        
        # 清理实体值（去除标点符号和多余空格）
        entity_clean = re.sub(r'[^\w\s]', '', entity.value).strip()
        entity_clean = re.sub(r'\s+', '', entity_clean)
        
        if not entity_clean:
            return None
        
        # 在原文附近区域搜索
        search_start = max(0, entity.start - self.max_position_offset)
        search_end = min(len(original_text), entity.end + self.max_position_offset)
        search_region = original_text[search_start:search_end]
        
        # 在搜索区域中查找清理后的实体值
        for i in range(len(search_region) - len(entity_clean) + 1):
            candidate = search_region[i:i + len(entity_clean)]
            candidate_clean = re.sub(r'[^\w\s]', '', candidate).strip()
            candidate_clean = re.sub(r'\s+', '', candidate_clean)
            
            if candidate_clean == entity_clean:
                absolute_start = search_start + i
                absolute_end = absolute_start + len(candidate)
                
                logger.debug(
                    f"模糊搜索成功: {entity.entity_type} "
                    f"位置修正从 ({entity.start}, {entity.end}) "
                    f"到 ({absolute_start}, {absolute_end})"
                )
                
                return EntityMatch(
                    entity_type=entity.entity_type,
                    value=candidate,
                    start=absolute_start,
                    end=absolute_end,
                    confidence=entity.confidence * 0.7,  # 显著降低置信度
                    is_inferred=entity.is_inferred,
                    notes=entity.notes,
                    metadata=entity.metadata
                )
        
        return None
    
    def _try_correct_position(
        self, 
        entity: EntityMatch, 
        original_text: str
    ) -> Optional[EntityMatch]:
        """
        尝试修正实体的位置信息，使用多种搜索策略
        
        Args:
            entity: 原始实体
            original_text: 原始文本
            
        Returns:
            Optional[EntityMatch]: 修正后的实体，如果无法修正则返回None
        """
        search_value = entity.value.strip()
        
        # 策略1：在全文中查找精确匹配
        start_pos = original_text.find(search_value)
        if start_pos != -1:
            end_pos = start_pos + len(search_value)
            logger.debug(
                f"精确匹配修正: {entity.entity_type} "
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
        
        # 策略2：在原位置附近搜索（允许一定偏移）
        search_start = max(0, entity.start - self.max_position_offset)
        search_end = min(len(original_text), entity.end + self.max_position_offset)
        search_region = original_text[search_start:search_end]
        
        # 在搜索区域中查找
        relative_pos = search_region.find(search_value)
        if relative_pos != -1:
            absolute_start = search_start + relative_pos
            absolute_end = absolute_start + len(search_value)
            
            logger.debug(
                f"附近搜索修正: {entity.entity_type} "
                f"从 ({entity.start}, {entity.end}) 到 ({absolute_start}, {absolute_end})"
            )
            
            return EntityMatch(
                entity_type=entity.entity_type,
                value=search_value,
                start=absolute_start,
                end=absolute_end,
                confidence=entity.confidence * 0.9,  # 略微降低置信度
                is_inferred=entity.is_inferred,
                notes=entity.notes,
                metadata=entity.metadata
            )
        
        # 策略3：查找所有匹配位置，选择最接近的
        all_positions = []
        start = 0
        while True:
            pos = original_text.find(search_value, start)
            if pos == -1:
                break
            all_positions.append(pos)
            start = pos + 1
        
        if all_positions:
            # 选择距离原位置最近的匹配
            closest_pos = min(all_positions, key=lambda x: abs(x - entity.start))
            closest_end = closest_pos + len(search_value)
            
            logger.debug(
                f"最近匹配修正: {entity.entity_type} "
                f"从 ({entity.start}, {entity.end}) 到 ({closest_pos}, {closest_end})"
            )
            
            return EntityMatch(
                entity_type=entity.entity_type,
                value=search_value,
                start=closest_pos,
                end=closest_end,
                confidence=entity.confidence * 0.8,  # 更多降低置信度
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
