# core/agent_tools.py
from typing import Annotated, List, Optional
from pathlib import Path
from langchain_core.tools import tool
from langchain.tools import InjectedState
from langchain_core.documents import Document
from langgraph.types import interrupt

# 导入你现有的核心类
from core.rag_chain import SelfRAGChain
from config.setting import RAGConfig
from core.agent_schemas import (
    RetrieveInput, EvaluateInput, OCRInput,
    HumanReviewInput, SystemStatusInput
)


# 全局上下文（替代依赖注入）
class ToolContext:
    """工具上下文：存储RAG链和配置的引用"""

    def __init__(self):
        self.rag_chain: Optional[SelfRAGChain] = None
        self.config: Optional[RAGConfig] = None

    def init(self, rag: SelfRAGChain, cfg: RAGConfig):
        """初始化（在Agent启动时调用）"""
        self.rag_chain = rag
        self.config = cfg


# 创建全局实例
ctx = ToolContext()


def init_agent_context(rag: SelfRAGChain, cfg: RAGConfig):
    """供Agent初始化的接口"""
    ctx.init(rag, cfg)


@tool(args_schema=RetrieveInput, response_format="content_and_artifact")
async def retrieve_knowledge(
        query: str,
        top_k: int = 3,
        use_rerank: bool = True,
        state: Annotated[Optional[dict], InjectedState] = None  # 类型改为 Optional[dict]
) -> tuple[str, List[Document]]:
    """
    从知识库检索相关文档。回答事实性问题前必须先调用此工具。
    返回格式化的文档摘要（给LLM）和原始Document对象（给评估器）。
    """
    if not ctx.rag_chain or not ctx.rag_chain.retriever:
        raise ValueError("知识库未初始化")

    # 检查迭代次数防止无限循环
    current_iter = state.get("iteration_count", 0) if state else 0
    if current_iter >= ctx.config.max_iterations:
        return "已达到最大检索迭代次数", []

    # 执行检索
    docs = await ctx.rag_chain.retriever.retrieve(query)

    # 格式化给LLM阅读的内容
    content_parts = [f"检索到 {len(docs)} 篇相关文档："]
    for i, doc in enumerate(docs[:top_k], 1):
        source = doc.metadata.get("source", "unknown")
        score = (
                doc.metadata.get("rerank_score") or
                doc.metadata.get("vector_score") or
                doc.metadata.get("bm25_score", 0)
        )
        preview = doc.page_content[:200].replace("\n", " ")
        content_parts.append(f"[{i}] {source}(相关度:{score:.2f}): {preview}...")

    return "\n".join(content_parts), docs[:top_k]


@tool(args_schema=EvaluateInput, response_format="content")
def evaluate_answer_quality(
        query: str,
        answer: str,
        contexts: List[str] = None,
        state: Annotated[Optional[dict], InjectedState] = None
) -> str:
    """
    评估生成答案的质量，检测幻觉风险和置信度。
    必须在生成答案后调用此工具进行自检。
    如果置信度低于阈值，系统会自动标记需要人工审核。
    """
    if not ctx.rag_chain:
        return "评估器未初始化"

    # 修复：变量名从 ctx 改为 text，避免遮蔽全局 ctx
    docs = [Document(page_content=text) for text in (contexts or [])]

    # 复用你现有的evaluator（避免重复代码）
    review = ctx.rag_chain.evaluator.evaluate(
        query=query,
        answer=answer,
        documents=docs,
        latency_ms=0
    )

    # 构建详细报告
    lines = [
        "【Self-RAG评估报告】",
        f"置信度: {review.confidence:.0%} (阈值: {ctx.config.human_review_threshold:.0%})",
        f"幻觉风险: {review.hallucination_risk:.0%} {'⚠️高风险' if review.hallucination_risk > 0.5 else '✅正常'}",
        f"检索相关性: {review.retrieval_relevance:.2f}",
        f"完整性: {review.answer_completeness:.2f}"
    ]

    # 如果触发审核条件，在State中标记（供Graph节点读取）
    if review.needs_human_review and state is not None:
        lines.append(f"\n⚠️ 触发人工审核条件，任务ID将生成")

    return "\n".join(lines)


@tool(args_schema=OCRInput, response_format="content")
async def process_document(
        file_path: str,
        language: str = "chi_sim+eng",
        auto_index: bool = True,
        state: Annotated[Optional[dict], InjectedState] = None
) -> str:
    """
    处理上传的图片或PDF文档，提取文字内容。
    当用户上传文件并询问其中内容时使用此工具。
    """
    from core.ocr_processor import OCRProcessor

    processor = OCRProcessor(language=language, enabled=True)
    if not processor.is_available():
        return "❌ OCR功能不可用（请安装Tesseract: apt-get install tesseract-ocr-chi-sim）"

    try:
        path = Path(file_path)
        if not path.exists():
            return f"❌ 文件不存在: {file_path}"

        text = await processor.extract_text(path)

        if not text:
            return "⚠️ 未能从文件中识别到文字内容"

        # 自动索引到知识库（保持数据新鲜度）
        if auto_index and ctx.rag_chain:
            await ctx.rag_chain.aindex_documents(
                texts=[text],
                metadatas=[{
                    "source": f"upload_{path.name}",
                    "type": "ocr_document",
                    "original_path": str(path)
                }]
            )
            index_info = f"（已自动索引到知识库）"
        else:
            index_info = ""

        return f"✅ OCR识别成功{index_info}，共{len(text)}字符：\n\n{text[:800]}{'...' if len(text) > 800 else ''}"

    except Exception as e:
        return f"❌ 处理失败: {str(e)}"


@tool(args_schema=HumanReviewInput, response_format="content")
def trigger_human_review(
        reason: str,
        suggestion: Optional[str] = None,  # 这里也要 Optional
        state: Annotated[Optional[dict], InjectedState] = None
) -> str:
    """
    触发人工审核流程。当评估显示置信度过低或检测到高风险时，
    调用此工具暂停执行并等待人工介入。
    """
    if not ctx.rag_chain:
        return "系统未初始化"

    # 生成任务ID
    import uuid
    task_id = f"review_{uuid.uuid4().hex[:8]}"

    # 获取当前对话上下文（从State中）
    last_answer = ""
    if state:  # 添加检查避免 state 为 None
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, 'content') and not hasattr(msg, 'tool_calls'):
                last_answer = msg.content
                break

    # 存储到审核队列（复用你现有的review_tasks机制）
    ctx.rag_chain.review_tasks[task_id] = {
        "task_id": task_id,
        "query": state.get("last_query", "unknown") if state else "unknown",
        "original_answer": last_answer,
        "status": "pending",
        "reason": reason,
        "suggestion": suggestion
    }

    # 使用 interrupt 暂停执行（LangGraph 1.0+ API）
    result = interrupt({
        "type": "human_review_required",
        "task_id": task_id,
        "reason": reason,
        "original_answer": last_answer,
        "available_actions": ["approved", "rejected", "modified"]
    })

    # resume后返回结果
    action = result.get("action", "unknown")
    return f"✅ 人工审核完成: {action}"


@tool(args_schema=SystemStatusInput, response_format="content")
def check_system_status(
        detail: bool = False,
        state: Annotated[Optional[dict], InjectedState] = None
) -> str:
    """
    查询Self-RAG系统运行状态和配置信息。
    用于运维检查或向用户展示系统概况。
    """
    if not ctx.rag_chain or not ctx.config:
        return "系统尚未初始化"

    cfg = ctx.config
    rag = ctx.rag_chain

    lines = [
        "📊 Self-RAG系统状态",
        f"• LLM模型: {cfg.llm_model}",
        f"• Embedding模型: {cfg.embedding_model}",
        f"• 文档总数: {len(rag.retriever.hybrid_retriever.documents) if rag.retriever else 0}",
        f"• 混合检索: {'BM25 + Vector' if cfg.hybrid_weights else '仅Vector'}",
        f"• 重排序: {'已启用' if cfg.reranker_enabled else '已禁用'}",
        f"• 人工审核: {'已启用' if cfg.human_review_enabled else '已禁用'}",
        f"• 待审核任务: {len([t for t in rag.review_tasks.values() if t['status'] == 'pending'])}"
    ]

    if detail:
        lines.extend([
            f"\n⚙️ 详细配置:",
            f"• 分块大小: {cfg.chunk_size}",
            f"• 重叠长度: {cfg.chunk_overlap}",
            f"• 最大迭代: {cfg.max_iterations}",
            f"• 置信度阈值: {cfg.confidence_threshold}",
            f"• OCR状态: {'已启用' if cfg.ocr_enabled else '已禁用'}"
        ])

    return "\n".join(lines)