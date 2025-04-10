import logging
from typing import List, Optional, Dict, Any

# 从 presidio_analyzer 导入核心类
from presidio_analyzer import EntityRecognizer, RecognizerResult, AnalysisExplanation
# 从我们的自定义模块导入所需的组件
from .llm_clients.base_client import BaseLLMClient # 导入 LLM 客户端基类
from .parsers.base_parser import BaseParser        # 导入解析器基类
# (可选) 导入 Prompt 加载和格式化工具
# from .prompts.prompt_loader import load_and_format_prompt

# 设置日志记录
logger = logging.getLogger(__name__)

class LLMRecognizer(EntityRecognizer):
    """
    一个使用大型语言模型 (LLM) 来识别 PII 实体的 Presidio Recognizer。

    该 Recognizer 通过向配置的 LLM 发送包含输入文本和检测指令的 Prompt 来工作。
    然后，它使用配置的解析器处理 LLM 的响应，以提取识别出的 PII 实体。
    """

    # 定义这个 Recognizer 支持的 PII 实体类型
    # 这些通常会在配置中定义，并传递给构造函数
    # 例如: ["PERSON", "PHONE_NUMBER", "CREDIT_CARD"]
    ENTITIES = [] # 稍后将在 __init__ 中设置

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
        prompt_template: str, # TODO 可以扩展为一个更复杂的 Prompt 管理对象
        prompt_format_vars: Optional[Dict[str, Any]] = None, # Prompt 模板中所需的额外变量
        name: Optional[str] = None,
        supported_language: str = "zh", # 默认支持中文
        version: Optional[str] = None,
        context: Optional[List[str]] = None,
        score_threshold: float = 0.5 # (可选) 可以根据 LLM 输出的置信度过滤
    ):
        """
        初始化 LLMRecognizer。

        Args:
            supported_entities (List[str]): 此 Recognizer 实例应识别的实体列表 (例如 ["PERSON", "EMAIL"])。
            llm_client (BaseLLMClient): 用于与 LLM API 交互的客户端实例 (继承自 BaseLLMClient)。
            parser (BaseParser): 用于解析 LLM 响应的解析器实例 (继承自 BaseParser)。
            prompt_template (str): 用于生成 LLM 请求的 Prompt 模板字符串或标识符。(未来可以扩展为一个更复杂的 Prompt 管理对象)
            prompt_format_vars (Optional[Dict[str, Any]]): (可选) 传递给 Prompt 格式化器的额外变量字典。
            name (Optional[str]): (可选) Recognizer 的名称。默认为 "LLMRecognizer"。
            supported_language (str): (可选) 此 Recognizer 支持的主要语言。默认为 "zh"。
            version (Optional[str]): (可选) Recognizer 的版本。默认为 "0.1.0"。
            context (Optional[List[str]]): (可选) 提供给 Recognizer 的上下文词列表。
            score_threshold (float): (可选) 低于此分数的识别结果将被忽略。
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

        # 加载或准备 Recognizer 可能需要的资源（例如，加载 Prompt 文件）
        self.load()

    def load(self) -> None:
        """
        加载 Recognizer 所需的资源。
        对于 LLM Recognizer，这可能涉及加载 Prompt 文件内容，
        或者确保 LLM 客户端已正确配置。
        """
        logger.info(f"加载 {self.name} (版本: {self.version}) for language {self.supported_language}...")

        # 示例：如果 prompt_template 是文件路径，可以在这里加载
        # try:
        #     # 假设 prompt_loader.py 中有 load_prompt_from_file 函数
        #     self.prompt_content = load_prompt_from_file(self.prompt_template)
        #     logger.info(f"成功从 {self.prompt_template} 加载 Prompt。")
        # except Exception as e:
        #     logger.error(f"加载 Prompt 文件 {self.prompt_template} 失败: {e}")
        #     # 根据需要处理错误，可能抛出异常或设置默认 Prompt
        #     self.prompt_content = "默认的 PII 检测 Prompt: 在以下文本中查找 PII：{text}" # 示例默认值

        # 目前假设 prompt_template 就是字符串模板本身
        # TODO 替换为Prompt对象
        self.prompt_content = self.prompt_template
        logger.info(f"使用 Prompt 模板: {self.prompt_content[:100]}...") # 打印部分模板以供调试

        # 可以在这里添加检查 LLM 客户端是否准备就绪的逻辑
        if not self.llm_client:
             raise ValueError("LLM 客户端未初始化。")
        if not self.parser:
             raise ValueError("解析器未初始化。")

        logger.info(f"{self.name} 加载完成。")


    def analyze(
        self,
        text: str,
        entities: List[str],
        nlp_artifacts: Optional[Any] = None, # 可以接收来自 NLP 引擎 (如 spaCy) 的额外信息
        language: Optional[str] = None, # 明确指定当前文本的语言
    ) -> List[RecognizerResult]:
        """
        使用 LLM 分析文本以查找指定的 PII 实体。

        Args:
            text (str): 要分析的输入文本。
            entities (List[str]): 要在此次分析中查找的实体类型列表。
                         注意：这会覆盖初始化时设置的 supported_entities，
                         允许更灵活的按需检测。
            nlp_artifacts: (可选) 来自上游 NLP 引擎的分析结果 (例如词性、依赖关系)。
                                LLM 可能不直接使用，但可以用于构建更复杂的 Prompt。
            language (Optional[str]): (可选) 文本的语言代码 (例如 'en', 'zh')。

        Returns: 一个 RecognizerResult 对象列表，包含找到的每个 PII 实体的信息。
        """
        logger.debug(f"开始使用 {self.name} 分析文本...")
        results: List[RecognizerResult] = []

        # 0. 确定要检测的实体
        # 如果调用 analyze 时传入了 entities 列表，则使用该列表；否则使用 Recognizer 初始化时支持的实体。
        # 还需要确保请求的实体是此 Recognizer 支持的。
        current_entities = [ent for ent in entities if ent in self.supported_entities]
        if not current_entities:
            logger.warning("没有请求的实体被此 Recognizer 支持，跳过分析。")
            return results # 如果没有要找的实体，直接返回空列表

        effective_language = language or self.supported_language
        logger.debug(f"语言: {effective_language}, 查找实体: {current_entities}")

        try:
            # 1. 准备 Prompt
            # 使用 prompt_content 模板和当前分析的上下文来格式化 Prompt
            # 需要传入文本本身、要查找的实体列表等信息
            prompt_input_vars = {
                "text": text,
                "entities": ", ".join(current_entities), # 将实体列表格式化为字符串
                **self.prompt_format_vars # 合并初始化时传入的额外变量
            }
            # 示例：假设 prompt_content 是 "请在以下文本中查找 {entities}: {text}"
            formatted_prompt = self.prompt_content.format(**prompt_input_vars)
            logger.debug(f"格式化后的 Prompt (前 200 字符): {formatted_prompt[:200]}...")

            # 2. 调用 LLM 客户端
            logger.debug("向 LLM 发送请求...")
            llm_response = self.llm_client.generate(prompt=formatted_prompt)
            logger.debug(f"收到 LLM 响应 (前 200 字符): {llm_response[:200]}...")

            # 3. 解析 LLM 响应
            # 使用配置的 parser 来解析 LLM 返回的原始文本或结构化数据
            logger.debug("开始解析 LLM 响应...")
            # 解析器需要知道原始文本，以便计算实体的位置 (start, end)
            parsed_results = self.parser.parse(llm_response, original_text=text, entities=current_entities)
            logger.debug(f"解析得到 {len(parsed_results)} 个潜在结果。")

            # 4. 将解析结果转换为 Presidio RecognizerResult 对象
            for p_res in parsed_results:
                # p_res 应该是一个包含 'entity_type', 'start', 'end', 'score' (可选) 的字典或对象
                entity_type = p_res.get("entity_type")
                start = p_res.get("start")
                end = p_res.get("end")
                score = p_res.get("score", 1.0) # 如果解析器不提供分数，默认为 1.0
                identified_text = text[start:end] # 从原始文本中提取识别出的文本

                # 检查解析结果是否有效且符合要求
                if entity_type in current_entities and start is not None and end is not None:
                    # 应用分数阈值过滤
                    if score >= self.score_threshold:
                        # 创建 AnalysisExplanation (可选，但推荐)
                        explanation = self.build_explanation(
                            original_score=score,
                            validation_result=None, # 如果有验证逻辑，可以在这里添加结果
                            recognizer_name=self.name,
                            pattern_name=None, # LLM 通常没有固定模式名
                            pattern=None,      # LLM 通常没有固定模式
                        )

                        # 创建 RecognizerResult
                        result = RecognizerResult(
                            entity_type=entity_type,
                            start=start,
                            end=end,
                            score=score,
                            analysis_explanation=explanation,
                            recognition_metadata={ # 可以添加 LLM 特有的元数据
                                "model_name": self.llm_client.model_name if hasattr(self.llm_client, 'model_name') else 'unknown',
                                "prompt_template_id": self.prompt_template, # 记录使用的模板
                            }
                        )
                        results.append(result)
                        logger.debug(f"添加结果: {result}")
                    else:
                         logger.debug(f"结果分数 {score} 低于阈值 {self.score_threshold}，已忽略: {identified_text} ({entity_type})")
                else:
                    logger.warning(f"解析器返回了无效或不匹配的结果: {p_res}")

        except Exception as e:
            # 处理 LLM 调用或解析过程中的错误
            logger.error(f"在 {self.name} 分析过程中发生错误: {e}", exc_info=True)
            # 根据策略，可以选择返回空列表或重新抛出异常

        logger.info(f"{self.name} 分析完成，找到 {len(results)} 个结果。")
        return results

    def build_explanation(
        self,
        original_score: float,
        validation_result: Optional[bool] = None,
        recognizer_name: Optional[str] = None,
        pattern_name: Optional[str] = None,
        pattern: Optional[str] = None,
    ) -> AnalysisExplanation:
        """
        为 LLM 的识别结果构建 AnalysisExplanation 对象。

        Args:
            original_score (float): LLM 或解析器提供的原始分数。
            validation_result (Optional[bool]): 验证步骤的结果。
            recognizer_name (Optional[str]): Recognizer 的名称。
            pattern_name (Optional[str]): 模式名称 (LLM 通常没有)。
            pattern (Optional[str]): 模式本身 (LLM 通常没有)。
            
        Returns: AnalysisExplanation 对象。
        """
        explanation = AnalysisExplanation(
            recognizer=recognizer_name or self.name,
            original_score=original_score,
            score=original_score, # 可以根据验证结果调整分数
            textual_explanation=f"Identified as {self.name} using LLM with score {original_score:.2f}", # 提供简单的文本解释
            pattern_name=pattern_name,
            pattern=pattern,
            validation_result=validation_result,
            # 可以添加更多 LLM 特有的解释信息
            # supportiv_context_word: Optional[str] = None, # 也可以尝试从 LLM 获取
        )
        return explanation

    # EntityRecognizer 需要实现的属性方法
    @property
    def supported_entities(self) -> List[str]:
        """返回此 Recognizer 支持的实体列表。"""
        return self._supported_entities
