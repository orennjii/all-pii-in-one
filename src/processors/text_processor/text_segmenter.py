"""
文本分割器

此模块提供将文本分割为句子或段落的功能。
"""

import re
from typing import List, Optional


class TextSegmenter:
    """
    文本分割器
    用于将文本分割为句子或段落等更小的单位
    """
    
    def __init__(self, **kwargs):
        """
        初始化文本分割器
        
        Args:
            **kwargs: 额外参数
        """
        pass
        
    def segment(self, text: str, by: str = 'sentence', **kwargs) -> List[str]:
        """
        分割文本
        
        Args:
            text: 要分割的文本
            by: 分割方式，'sentence'表示按句子分割，'paragraph'表示按段落分割
            **kwargs: 额外参数
            
        Returns:
            List[str]: 分割后的文本片段列表
        """
        if not text:
            return []
            
        if by == 'sentence':
            return self.segment_by_sentence(text, **kwargs)
        elif by == 'paragraph':
            return self.segment_by_paragraph(text, **kwargs)
        else:
            return [text]
            
    def segment_by_sentence(self, text: str, **kwargs) -> List[str]:
        """
        按句子分割文本
        
        Args:
            text: 要分割的文本
            **kwargs: 额外参数
            
        Returns:
            List[str]: 句子列表
        """
        # 中文句子分割模式
        # 匹配句号、问号、感叹号等句末标点
        pattern = r'([。！？；?!;])+|(\n\s*\n)'
        
        # 分割文本
        segments = re.split(pattern, text)
        
        # 过滤None和空字符串
        segments = [s for s in segments if s and not s.isspace()]
        
        # 如果分割结果为空，则返回原文本
        if not segments:
            return [text]
            
        # 重新组合句子与标点
        sentences = []
        current = ""
        
        for segment in segments:
            if re.match(r'([。！？；?!;])+|(\n\s*\n)', segment):
                # 如果是标点或换行，则添加到当前句子并结束句子
                if current:
                    sentences.append(current + segment)
                    current = ""
            else:
                # 如果是文本内容，则添加到当前句子
                current += segment
                
        # 添加最后一个句子（如果有）
        if current:
            sentences.append(current)
            
        return sentences or [text]
        
    def segment_by_paragraph(self, text: str, **kwargs) -> List[str]:
        """
        按段落分割文本
        
        Args:
            text: 要分割的文本
            **kwargs: 额外参数
            
        Returns:
            List[str]: 段落列表
        """
        # 按连续两个换行符分割
        paragraphs = re.split(r'\n\s*\n', text)
        
        # 过滤空段落
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        return paragraphs or [text]
