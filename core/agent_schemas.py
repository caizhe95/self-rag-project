# core/agent_schemas.py
from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class RetrieveInput(BaseModel):
    """检索知识库的输入参数"""
    query: str = Field(
        description="检索查询语句，应提取用户问题的核心实体和关键词",
        min_length=1,
        max_length=500
    )
    top_k: int = Field(
        default=3,
        description="返回的文档数量（1-10），小模型建议3，大模型可6",
        ge=1,
        le=10
    )
    use_rerank: bool = Field(
        default=True,
        description="是否启用Cross-Encoder重排序提升精度"
    )


class EvaluateInput(BaseModel):
    """评估答案质量的输入参数"""
    query: str = Field(
        description="原始用户问题，用于判断答案是否回答问题本身"
    )
    answer: str = Field(
        description="生成的答案内容，需要被评估的文本"
    )
    contexts: List[str] = Field(
        default=[],
        description="检索到的参考文档内容列表，用于验证答案事实性"
    )


class OCRInput(BaseModel):
    """OCR文档处理的输入参数"""
    file_path: str = Field(
        description="待处理文件的绝对路径（支持.jpg/.png/.pdf）"
    )
    language: str = Field(
        default="chi_sim+eng",
        description="OCR语言包（中文chi_sim，英文eng，混合chi_sim+eng）"
    )
    auto_index: bool = Field(
        default=True,
        description="识别后是否自动索引到知识库"
    )


class HumanReviewInput(BaseModel):
    """触发人工审核的输入参数"""
    reason: str = Field(
        description="触发审核的具体原因（如'置信度0.3低于阈值'）"
    )
    suggestion: Optional[str] = Field(
        default=None,
        description="给审核员的建议或备注"
    )


class SystemStatusInput(BaseModel):
    """查询系统状态的输入参数"""
    detail: bool = Field(
        default=False,
        description="是否显示详细配置（分块大小、模型参数等）"
    )