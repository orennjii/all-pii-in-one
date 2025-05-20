"""
文本匿名化引擎

此模块提供实际替换文本中的PII的功能。
支持多种匿名化策略，如替换、掩码、假数据等。
"""

import re
import uuid
import random
import string
from typing import Dict, List, Tuple, Optional, Union, Any

from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import RecognizerResult, OperatorConfig


class TextAnonymizerEngine:
    """
    文本匿名化引擎
    负责将文本中检测到的PII进行匿名化处理
    """
    
    def __init__(self, **kwargs):
        """
        初始化文本匿名化引擎
        
        Args:
            **kwargs: 额外参数
        """
        # 初始化presidio匿名化引擎
        self.presidio_anonymizer = AnonymizerEngine()
        
        # 默认匿名化策略
        self.default_strategies = {
            'PERSON': 'replace',
            'PHONE_NUMBER': 'mask',
            'EMAIL_ADDRESS': 'mask',
            'ADDRESS': 'replace',
            'ID_CARD': 'mask',
            'BANK_CARD': 'mask',
            'IP_ADDRESS': 'mask',
            'LOCATION': 'replace',
            'ORGANIZATION': 'replace',
            'DEFAULT': 'mask'
        }
        
        # 替换值生成器
        self.replacers = {
            'PERSON': self._generate_fake_name,
            'ADDRESS': self._generate_fake_address,
            'LOCATION': self._generate_fake_location,
            'ORGANIZATION': self._generate_fake_organization,
            'DEFAULT': lambda value, **kwargs: f"[{value}]"
        }
        
    def anonymize(
        self, 
        text: str, 
        pii_results: List[Dict], 
        strategy: Dict = None
    ) -> Tuple[str, Dict[str, str]]:
        """
        匿名化文本
        
        Args:
            text: 要匿名化的文本
            pii_results: PII检测结果
            strategy: 匿名化策略,如{'PERSON': 'replace', 'PHONE_NUMBER': 'mask'}
            
        Returns:
            Tuple[str, Dict]: 匿名化后的文本和PII映射关系
        """
        if not text or not pii_results:
            return text, {}
            
        # 合并默认策略和用户指定策略
        anonymization_strategy = self.default_strategies.copy()
        if strategy:
            anonymization_strategy.update(strategy)
            
        # 按位置逆序排序,以便从后向前替换,避免位置错误
        sorted_results = sorted(pii_results, key=lambda x: x['start'], reverse=True)
        
        # 保存原始PII和替换值的映射
        pii_mapping = {}
        
        # 执行匿名化
        anonymized_text = text
        for result in sorted_results:
            entity_type = result['entity_type']
            start = result['start']
            end = result['end']
            original_value = result['value']
            
            # 获取该类型的匿名化策略
            method = anonymization_strategy.get(entity_type, anonymization_strategy.get('DEFAULT', 'mask'))
            
            # 根据策略执行匿名化
            if method == 'mask':
                # 掩码处理: ******
                replacement = '*' * len(original_value)
            elif method == 'redact':
                # 编辑处理: [已编辑]
                replacement = f"[已编辑]"
            elif method == 'hash':
                # 哈希处理: 产生一个固定哈希值
                replacement = str(hash(original_value))[:8]
            elif method == 'replace':
                # 替换处理: 使用替换生成器
                replacer = self.replacers.get(entity_type, self.replacers['DEFAULT'])
                replacement = replacer(original_value, entity_type=entity_type)
            else:
                # 默认掩码
                replacement = '*' * len(original_value)
                
            # 执行替换
            anonymized_text = anonymized_text[:start] + replacement + anonymized_text[end:]
            
            # 保存映射关系
            pii_mapping[replacement] = original_value
            
        return anonymized_text, pii_mapping
        
    def restore(self, anonymized_text: str, pii_mapping: Dict[str, str]) -> str:
        """
        恢复匿名化文本到原始文本
        
        Args:
            anonymized_text: 匿名化后的文本
            pii_mapping: PII映射关系
            
        Returns:
            str: 恢复后的文本
        """
        if not anonymized_text or not pii_mapping:
            return anonymized_text
            
        # 按照长度逆序排序键,以避免部分匹配问题
        sorted_keys = sorted(pii_mapping.keys(), key=len, reverse=True)
        
        restored_text = anonymized_text
        for key in sorted_keys:
            original = pii_mapping[key]
            restored_text = restored_text.replace(key, original)
            
        return restored_text
        
    def _generate_fake_name(self, value: str, **kwargs) -> str:
        """
        生成假名字
        
        Args:
            value: 原始值
            **kwargs: 额外参数
            
        Returns:
            str: 假名字
        """
        fake_names = ['张三', '李四', '王五', '赵六', '钱七', '孙八', '周九', '吴十']
        return random.choice(fake_names)
        
    def _generate_fake_address(self, value: str, **kwargs) -> str:
        """
        生成假地址
        
        Args:
            value: 原始值
            **kwargs: 额外参数
            
        Returns:
            str: 假地址
        """
        fake_addresses = ['北京市海淀区某某路123号', '上海市浦东新区某某街456号', 
                          '广州市天河区某某大道789号', '深圳市南山区某某园区10号']
        return random.choice(fake_addresses)
        
    def _generate_fake_location(self, value: str, **kwargs) -> str:
        """
        生成假位置
        
        Args:
            value: 原始值
            **kwargs: 额外参数
            
        Returns:
            str: 假位置
        """
        fake_locations = ['北京', '上海', '广州', '深圳', '杭州', '南京', '成都', '重庆']
        return random.choice(fake_locations)
        
    def _generate_fake_organization(self, value: str, **kwargs) -> str:
        """
        生成假组织名称
        
        Args:
            value: 原始值
            **kwargs: 额外参数
            
        Returns:
            str: 假组织名称
        """
        fake_orgs = ['某某科技有限公司', '某某集团', '某某大学', '某某研究所', '某某局']
        return random.choice(fake_orgs)
