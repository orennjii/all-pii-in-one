"""
LLM识别器模块。
"""
import logging
from typing import List, Optional, Dict, Any

from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation

from .clients.base_client import BaseLLMClient
from .parsers.base_parser import BaseParser

# 设置日志记录
logger = logging.getLogger(__name__)


class LLMRecognizer(EntityRecognizer):
    """
    一个使用大型语言模型 (LLM) 来识别 PII 实体的 Presidio Recognizer。

    该 Recognizer 通过向配置的 LLM 发送包含输入文本和检测指令的 Prompt 来工作。
    然后，它使用配置的解析器处理 LLM 的响应，以提取识别出的 PII 实体。
    """

    # 定义这个 Recognizer 支持的 PII 实体类型
    ENTITIES = []  # 稍后将在 __init__ 中设置

    NAME = "LLMRecognizer"

    SUPPORTED_LANGUAGES = ["zh"]

    VERSION = "0.1.0"

    # 定义此 Recognizer 的上下文词（如果需要，用于提高准确性）
    # 对于 LLM，上下文通常由 Prompt 处理，这里可能不需要
    CONTEXT = []

    def __init__(
        self,
        supported_entities: List[str],
        llm_client: BaseLLMClient,
        parser: BaseParser,
        prompt_template: str,  # TODO 可以扩展为一个更复杂的 Prompt 管理对象
        prompt_format_vars: Optional[Dict[str, Any]] = None,  # Prompt 模板中所需的额外变量
        name: Optional[str] = None,
        supported_language: str = "zh",  # 默认支持中文
        version: Optional[str] = None,
        context: Optional[List[str]] = None,
        score_threshold: float = 0.5  # (可选) 可以根据 LLM 输出的置信度过滤
    ):
        """
        初始化 LLMRecognizer。

        Args:
            supported_entities: 此 Recognizer 实例应识别的实体列表 (例如 ["PERSON", "EMAIL"])
            llm_client: 用于与 LLM API 交互的客户端实例 (继承自 BaseLLMClient)
            parser: 用于解析 LLM 响应的解析器实例 (继承自 BaseParser)
            prompt_template: 用于生成 LLM 请求的 Prompt 模板字符串或标识符
            prompt_format_vars: (可选) 传递给 Prompt 格式化器的额外变量字典
            name: (可选) Recognizer 的名称。默认为 "LLMRecognizer"
            supported_language: (可选) 此 Recognizer 支持的主要语言。默认为 "zh"
            version: (可选) Recognizer 的版本。默认为 "0.1.0"
            context: (可选) 提供给 Recognizer 的上下文词列表
            score_threshold: (可选) 低于此分数的识别结果将被忽略
        """
        # 设置实例属性
        self.llm_client = llm_client
        self.parser = parser
        self.prompt_template = prompt_template
        self.prompt_format_vars = prompt_format_vars if prompt_format_vars else {}
        self.score_threshold = score_threshold

        # 设置 Presidio EntityRecognizer 所需的属性
        _supported_entities = supported_entities or self.ENTITIES
        _name = name or self.NAME
        _supported_language = supported_language or self.SUPPORTED_LANGUAGES[0]
        _version = version or self.VERSION
        _context = context or self.CONTEXT

        # 调用父类的构造函数
        super().__init__(
            supported_entities=_supported_entities,
            name=_name,
            supported_language=_supported_language,
            version=_version,
            context=_context
        )

    def load(self) -> None:
        """
        加载 Recognizer 所需的资源。
        对于 LLM Recognizer，这可能涉及加载 Prompt 文件内容，
        或者确保 LLM 客户端已正确配置。
        """
        logger.info(f"加载 {self.name} (版本: {self.version}) for language "
                    f"{self.supported_language}...")
        # 这里可以添加其他初始化操作

    def analyze(self, text: str, entities: List[str] = None, nlp_artifacts=None) -> List[RecognizerResult]:
        """
        分析文本以识别受支持的 PII 实体。

        Args:
            text: 要分析的文本
            entities: 要识别的实体类型列表 (如果 None，则使用 self.supported_entities)
            nlp_artifacts: 来自 NlpEngine 的可选预处理结果

        Returns:
            List[RecognizerResult]: 识别结果列表
        """
        # 如果提供了特定实体列表，则只识别这些实体
        if entities:
            # 只分析此识别器支持的实体
            entities_to_analyze = [
                entity for entity in entities if entity in self.supported_entities
            ]
            if not entities_to_analyze:
                return []
        else:
            entities_to_analyze = self.supported_entities

        # 格式化 Prompt
        formatted_prompt = self._format_prompt(
            text=text,
            entities=entities_to_analyze
        )

        # 调用 LLM
        try:
            llm_response = self.llm_client.generate(
                prompt=formatted_prompt,
                temperature=0.0  # 使用确定性输出以获得一致的结果
            )
        except Exception as e:
            logger.error(f"调用 LLM 时出错: {str(e)}")
            return []

        # 使用解析器处理响应
        try:
            recognizer_results = self.parser.parse(
                text=text,
                llm_response=llm_response,
                entities=entities_to_analyze
            )
        except Exception as e:
            logger.error(f"解析 LLM 响应时出错: {str(e)}")
            return []

        # 应用分数阈值过滤
        if self.score_threshold > 0:
            recognizer_results = [
                result for result in recognizer_results
                if result.score >= self.score_threshold
            ]

        return recognizer_results

    def _format_prompt(self, text: str, entities: List[str]) -> str:
        """
        格式化发送到 LLM 的 Prompt。

        Args:
            text: 要分析的文本
            entities: 要识别的实体类型列表

        Returns:
            str: 格式化后的 Prompt
        """
        # 简单的模板填充方法，可以根据需求扩展
        format_vars = {
            "text": text,
            "entities": ", ".join(entities),
            **self.prompt_format_vars  # 添加自定义变量
        }
        
        return self.prompt_template.format(**format_vars)
