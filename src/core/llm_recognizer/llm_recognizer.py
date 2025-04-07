import logging
from typing import List, Optional, Dict, Any, Tuple, Set
import json

from presidio_analyzer import EntityRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

from .llm_clients.base_client import BaseLLMClient
from .parsers.base_parser import BaseParser
from .config.settings import Settings

logger = logging.getLogger(__name__)

class LLMRecognizer(EntityRecognizer):
    """
    基于大语言模型的实体识别器，继承自Presidio的EntityRecognizer。
    
    该识别器使用大语言模型识别文本中的PII信息，通过：
    1. 初始化时配置LLM客户端、提示词、输出解析器
    2. 向LLM发送带有上下文的提示词
    3. 解析LLM的输出，转换为标准的RecognizerResult格式
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        parser: BaseParser,
        prompt_template: str,
        supported_entities: List[str],
        config: Optional[Dict[str, Any]] = None,
        supported_language: str = "en",
        name: str = "LLMRecognizer"
    ):
        """
        初始化LLM识别器。
        
        Args:
            llm_client: LLM客户端实例，用于与大语言模型交互
            parser: 用于解析LLM输出的解析器
            prompt_template: 提示词模板
            supported_entities: 此识别器支持识别的实体类型列表
            config: 额外配置选项
            supported_language: 支持的语言 (默认英语)
            name: 识别器名称
        """
        super().__init__(
            supported_entities=supported_entities,
            supported_language=supported_language,
            name=name
        )
        
        self.llm_client = llm_client
        self.parser = parser
        self.prompt_template = prompt_template
        self.config = config or {}
        
        # 设置默认分数
        self.default_score = self.config.get("default_score", 0.7)
        
        # 日志记录初始化
        logger.info(f"初始化 {name} 识别器，支持实体: {supported_entities}")
    
    def load(self) -> None:
        """
        加载识别器所需的资源。
        在LLM识别器中，这可能包括预热LLM客户端或验证连接。
        """
        logger.debug("加载LLM识别器资源")
        # 可选：在这里添加预热或验证步骤
        self.is_loaded = True
    
    def analyze(
        self, 
        text: str, 
        entities: List[str], 
        nlp_artifacts: Optional[NlpArtifacts] = None
    ) -> List[RecognizerResult]:
        """
        分析文本并返回检测到的实体列表。
        
        Args:
            text: 要分析的文本
            entities: 要查找的实体类型列表
            nlp_artifacts: 自然语言处理的结果对象(此实现可能不使用)
            
        Returns:
            List[RecognizerResult]: 识别到的实体结果列表
        """
        if not self.is_loaded:
            self.load()
            
        if not set(entities).intersection(set(self.supported_entities)):
            return []
            
        # 筛选当前Recognizer支持的实体
        relevant_entities = list(set(entities).intersection(set(self.supported_entities)))
        
        # 准备提示词
        prompt = self._prepare_prompt(text, relevant_entities)
        
        try:
            # 调用LLM
            llm_response = self.llm_client.generate(prompt)
            
            # 解析结果
            parsed_results = self.parser.parse(llm_response, text)
            
            # 转换为Presidio的RecognizerResult对象
            recognizer_results = self._convert_to_recognizer_results(parsed_results, relevant_entities)
            
            logger.debug(f"LLM识别器发现 {len(recognizer_results)} 个实体")
            return recognizer_results
            
        except Exception as e:
            logger.error(f"LLM分析过程出错: {str(e)}")
            return []
    
    def _prepare_prompt(self, text: str, entities: List[str]) -> str:
        """
        准备发送给LLM的提示词。
        
        Args:
            text: 要分析的文本
            entities: 要识别的实体类型
            
        Returns:
            str: 格式化后的提示词
        """
        # 基本的提示词格式化，这里使用简单的替换
        # 在实际应用中，可能会使用更复杂的提示词模板系统
        formatted_prompt = self.prompt_template.replace("{TEXT}", text)
        formatted_prompt = formatted_prompt.replace("{ENTITIES}", ", ".join(entities))
        
        return formatted_prompt
    
    def _convert_to_recognizer_results(
        self, 
        parsed_results: List[Dict[str, Any]], 
        relevant_entities: List[str]
    ) -> List[RecognizerResult]:
        """
        将解析后的LLM输出转换为标准的RecognizerResult对象列表。
        
        Args:
            parsed_results: 解析器输出的结果列表
            relevant_entities: 相关的实体类型
            
        Returns:
            List[RecognizerResult]: Presidio识别器结果列表
        """
        results = []
        
        for item in parsed_results:
            # 检查解析的结果是否包含所需的字段
            if not all(k in item for k in ["entity_type", "start", "end"]):
                logger.warning(f"跳过不完整的解析结果: {item}")
                continue
                
            # 检查实体类型是否在请求的实体列表中
            if item["entity_type"] not in relevant_entities:
                continue
                
            # 创建RecognizerResult对象
            score = item.get("score", self.default_score)
            result = RecognizerResult(
                entity_type=item["entity_type"],
                start=item["start"],
                end=item["end"],
                score=score
            )
            results.append(result)
            
        return results
    
    @classmethod
    def from_config(cls, config_path: Optional[str] = None):
        """
        从配置文件创建LLMRecognizer实例的工厂方法。
        
        Args:
            config_path: 配置文件路径，默认使用默认配置
            
        Returns:
            LLMRecognizer: 初始化好的识别器实例
        """
        # 加载配置
        settings = Settings(config_path)
        
        # 根据配置动态导入并实例化LLM客户端
        client_module = __import__(
            f"llm_recognizer.llm_clients.{settings.llm_client_type}_client", 
            fromlist=[""]
        )
        ClientClass = getattr(client_module, f"{settings.llm_client_type.capitalize()}Client")
        llm_client = ClientClass(**settings.llm_client_config)
        
        # 动态导入并实例化解析器
        parser_module = __import__(
            f"llm_recognizer.parsers.{settings.parser_type}_parser", 
            fromlist=[""]
        )
        ParserClass = getattr(parser_module, f"{settings.parser_type.capitalize()}Parser")
        parser = ParserClass(**settings.parser_config)
        
        # 加载提示词模板
        with open(settings.prompt_template_path, 'r', encoding='utf-8') as f:
            prompt_template = f.read()
        
        # 创建识别器实例
        return cls(
            llm_client=llm_client,
            parser=parser,
            prompt_template=prompt_template,
            supported_entities=settings.supported_entities,
            config=settings.recognizer_config,
            supported_language=settings.supported_language,
            name=settings.recognizer_name
        )