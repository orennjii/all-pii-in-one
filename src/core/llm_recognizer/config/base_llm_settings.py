from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

class BaseLLMSettings(BaseModel):
    """
    基础的 LLM 设置类，适用于所有 LLM Recognizer
    """

    prompt_file_path: str = Field(
        description="提示词文件的路径",
    )