"""
文本PII检测器

此模块提供用于检测文本中的个人隐私信息(PII)的接口和实现。
支持多种检测策略，包括基于presidio、基于模式和基于LLM的方法。
"""

from typing import Dict, List, Tuple, Optional, Union, Any
import re


class TextPIIDetector:
    """
    文本个人隐私信息(PII)检测器的基类
    提供检测文本中可能包含的个人隐私信息的接口
    
    支持组合多个检测器以提高检测效果
    """
    
    def __init__(self, detectors: Optional[List['TextPIIDetector']] = None) -> None:
        """
        初始化PII检测器
        
        Args:
            detectors: 子检测器列表,如果提供则会组合多个检测器的结果
        """
        self.detectors = detectors or []
        
    def detect(self, text: str, language: str = 'zh', **kwargs) -> List[Dict[str, Any]]:
        """
        检测文本中的PII信息
        
        Args:
            text: 要检测的文本
            language: 文本语言
            **kwargs: 额外的参数
            
        Returns:
            List[Dict]: 检测结果列表,每个结果是一个字典,包含以下字段:
                - entity_type: PII类型 (如PERSON, PHONE_NUMBER等)
                - start: 开始位置
                - end: 结束位置
                - value: 匹配到的内容
                - score: 置信度分数
        """
        if self.detectors:
            # 如果有子检测器,则组合所有子检测器的结果
            all_results = []
            for detector in self.detectors:
                all_results.extend(detector.detect(text, language, **kwargs))
            
            # 去除重叠结果
            return self._remove_overlapping_entities(all_results)
            
        # 基类的默认实现,子类需要重写此方法
        return []
    
    def _remove_overlapping_entities(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        去除重叠的实体,保留分数较高的
        
        Args:
            entities: 实体列表
            
        Returns:
            List[Dict]: 去除重叠后的实体列表
        """
        if not entities:
            return []
            
        # 按置信度分数降序排序
        sorted_entities = sorted(entities, key=lambda x: x.get('score', 0), reverse=True)
        
        result = []
        for current in sorted_entities:
            # 检查当前实体是否与已选择的实体重叠
            is_overlapping = False
            for selected in result:
                # 判断是否重叠
                if (current['start'] < selected['end'] and 
                    current['end'] > selected['start']):
                    is_overlapping = True
                    break
            
            # 如果不重叠,则添加到结果中
            if not is_overlapping:
                result.append(current)
                
        # 按位置排序
        return sorted(result, key=lambda x: x['start'])


class PresidioPIIDetector(TextPIIDetector):
    """
    基于Presidio的PII检测器
    使用Microsoft Presidio库进行PII检测
    """
    
    def __init__(self, analyzer=None) -> None:
        """
        初始化Presidio PII检测器
        
        Args:
            analyzer: Presidio的AnalyzerEngine实例
        """
        super().__init__()
        self.analyzer = analyzer
        
    def detect(self, text: str, language: str = 'zh', **kwargs) -> List[Dict[str, Any]]:
        """
        使用Presidio检测文本中的PII信息
        
        Args:
            text: 要检测的文本
            language: 文本语言
            **kwargs: 额外的参数
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if not text or not self.analyzer:
            return []
            
        # 调用Presidio进行检测
        analyzer_results = self.analyzer.analyze(
            text=text, 
            language=language,
            **kwargs
        )
        
        # 转换结果格式
        results = []
        for result in analyzer_results:
            results.append({
                'entity_type': result.entity_type,
                'start': result.start,
                'end': result.end,
                'value': text[result.start:result.end],
                'score': result.score,
                'analysis_explanation': getattr(result, 'analysis_explanation', None)
            })
            
        return results


class PatternPIIDetector(TextPIIDetector):
    """
    基于正则表达式模式的PII检测器
    使用预定义的正则表达式进行PII检测
    """
    
    def __init__(self) -> None:
        """
        初始化模式PII检测器
        """
        super().__init__()
        self.patterns = self._get_patterns()
        
    def _get_patterns(self) -> Dict[str, str]:
        """
        获取预定义的正则表达式模式
        
        Returns:
            Dict[str, str]: 模式字典,key为实体类型,value为正则表达式
        """
        # 这里只是一些简单示例,实际项目中可以增加更多模式
        return {
            'EMAIL': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
            'PHONE_NUMBER': r'\b(?:\+?86)?1[3-9]\d{9}\b',
            'IP_ADDRESS': r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
            'URL': r'\bhttps?://[^\s()<>]+(?:\([^\s()<>]+\)|[^\s`!()\[\]{};:\'".,<>?«»""''])+',
        }
        
    def detect(self, text: str, language: str = 'zh', **kwargs) -> List[Dict[str, Any]]:
        """
        使用正则表达式检测文本中的PII信息
        
        Args:
            text: 要检测的文本
            language: 文本语言
            **kwargs: 额外的参数
            
        Returns:
            List[Dict]: 检测结果列表
        """
        if not text:
            return []
            
        results = []
        for entity_type, pattern in self.patterns.items():
            for match in re.finditer(pattern, text):
                results.append({
                    'entity_type': entity_type,
                    'start': match.start(),
                    'end': match.end(),
                    'value': match.group(0),
                    'score': 1.0  # 正则匹配通常给予高置信度
                })
                
        return results


class LLMPIIDetector(TextPIIDetector):
    """
    基于大型语言模型(LLM)的PII检测器
    使用LLM进行PII检测,更适合复杂或上下文相关的PII
    """
    
    def __init__(self, llm_client=None) -> None:
        """
        初始化基于LLM的PII检测器
        
        Args:
            llm_client: LLM客户端
        """
        super().__init__()
        self.llm_client = llm_client
        
    def detect(self, text: str, language: str = 'zh', **kwargs) -> List[Dict[str, Any]]:
        """
        使用LLM检测文本中的PII信息
        
        Args:
            text: 要检测的文本
            language: 文本语言
            **kwargs: 额外的参数
            
        Returns:
            List[Dict]: 检测结果列表
        """
        # LLM检测需要具体实现
        # 这里是一个占位实现
        return []
