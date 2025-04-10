from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# TODO : 统一管理所有 LLM 的配置项
class BaseLLMSettings(BaseModel):
    """
    基础的 LLM 设置类，适用于所有 LLM Recognizer
    """

    # 基础配置
    model_name: str = Field(
        default = "",
        description = "LLM 模型名称",
    )
    prompt_template: str = Field(
        default = """
分析任务：请识别以下文本中的敏感信息实体。

支持的实体类型：
{supported_entities}

请以JSON格式返回结果，JSON结构为数组，每个项目包含以下字段和顺序：
- entity_type: 实体类型，必须是上述支持的类型之一
- start: 实体在原文中的起始位置索引
- end: 实体在原文中的结束位置索引
- score: 置信度分数，范围为0到1

示例输出格式：
[
{{"entity_type": "PERSON", "start": 5, "end": 7, "score": 0.95}},
{{"entity_type": "DATE_TIME", "start": 10, "end": 21, "score": 0.98}}
]

请分析以下文本：
{text}
        """,
        description = "LLM 提示模板",
    )
    supported_entities: List[str] = Field(
        default = ["DATE_TIME", "PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "IP_ADDRESS"],
        description = "支持的实体类型列表",
    )