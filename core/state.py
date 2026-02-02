# core/state.py
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document


class RAGState(TypedDict):
    """RAG 流程状态（面试亮点：支持多模态 + 人机协作）"""
    # 输入输出
    query: str
    chat_history: List[Dict[str, str]]

    # 文档与上下文
    documents: List[Document]
    context: str
    sources: List[Dict[str, Any]]

    # 答案与评估
    answer: str
    confidence: float
    iteration: int
    review_result: Optional[Dict[str, Any]]

    # 多模态支持
    ocr_texts: Optional[List[str]]
    images: Optional[List[bytes]]

    # 人机协作
    review_task_id: Optional[str]
    review_status: Optional[str]
    human_modified_answer: Optional[str]
    review_comment: Optional[str]
    reviewer: Optional[str]
    review_trigger_reason: Optional[str]

    # 历史上下文（记忆）
    history_context: str