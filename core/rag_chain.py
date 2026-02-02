# core/rag_chain.py
import asyncio
import time
import uuid
from typing import List, Dict, Any, Optional, Literal, Union
from pathlib import Path

from langchain_core.documents import Document
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

# ========== 关键导入：interrupt 和 Command ==========
from langgraph.types import Command, interrupt

from core.state import RAGState
from core.document_processor import DocumentProcessor
from core.retriever import Retriever
from core.generator import Generator
from evaluation.self_evaluator import SelfEvaluator
from config.setting import RAGConfig


class SelfRAGChain:
    """Self-RAG 主链：模块化 + LangGraph + 多模态就绪 + 人机协作（面试级实现）"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.review_tasks: Dict[str, Dict[str, Any]] = {}
        self.review_enabled = getattr(config, "human_review_enabled", False)

        # 初始化组件
        self.processor = DocumentProcessor(config)
        self.generator = Generator(
            ollama_base_url=config.ollama_base_url,
            llm_model=config.llm_model,
            temperature=config.temperature
        )
        self.evaluator = SelfEvaluator(self.generator.llm, config)

        self.retriever = None  # 初始为None，将在aindex_documents中设置
        self.review_tasks: Dict[str, Dict[str, Any]] = {}
        self.review_enabled = getattr(config, "human_review_enabled", False)

        # LangGraph Memory
        self.memory_store = InMemoryStore()
        self.checkpointer = InMemorySaver()
        self.graph = None
        self.session_id = "default"

    def enable_memory(self, session_id: str = "default"):
        """启用LangGraph Memory"""
        self.session_id = session_id

        if self.processor.vector_manager:
            embeddings = self.processor.vector_manager.embeddings
            self.memory_store = InMemoryStore(
                index={"embed": embeddings, "dims": 1024}
            )

        self.checkpointer = InMemorySaver()
        print(f"✅ LangGraph Memory 已启用 | 会话: {session_id}")

    async def aindex_documents(self, texts: List[str], files: Optional[List[Union[Path, str]]] = None,
                               metadatas: Optional[List[Dict]] = None):
        """索引文档（支持OCR）- 终极修复版"""
        if self.graph is not None:
            print("⚠️  图已存在，跳过重新编译")
            return

        try:
            # 1. 处理文档和OCR
            vector_store = await self.processor.process(texts, files, metadatas)

            # 2. 获取所有文档（包含OCR）
            documents = []
            if self.processor.vector_manager:
                documents = self.processor.vector_manager.get_all_documents()

            # 3. 确保至少有一个文档
            if not documents:
                print("⚠️  未获取到任何文档，使用空列表初始化")
                documents = []

            # 4. 初始化检索器
            self.retriever = Retriever(vector_store, documents, self.config)

            # 5. 构建图（确保graph被赋值）
            self.graph = self._build_graph()
            print(f"✅ StateGraph 编译完成 | 文档数：{len(documents)}")

        except Exception as e:
            print(f"❌ aindex_documents 失败: {e}")
            raise

    def _build_graph(self):
        """构建 Self-RAG 工作流"""
        graph = StateGraph(RAGState)

        # 添加节点（使用实例方法作为节点函数）
        graph.add_node("process_query", self._process_query_node)
        graph.add_node("retrieve", self._retrieve_node)
        graph.add_node("generate", self._generate_node)
        graph.add_node("evaluate", self._evaluate_node)
        graph.add_node("human_review", self._human_review_node)
        graph.add_node("refine", self._refine_node)
        graph.add_node("finalize", self._finalize_node)

        # 定义流程
        graph.add_edge(START, "process_query")
        graph.add_edge("process_query", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "evaluate")

        # 条件分支
        graph.add_conditional_edges("evaluate", self._should_continue, {
            "refine": "refine",
            "human_review": "human_review",
            "finalize": "finalize"
        })

        graph.add_conditional_edges("human_review", self._should_continue_after_review, {
            "refine": "refine",
            "finalize": "finalize"
        })

        graph.add_edge("refine", "retrieve")
        graph.add_edge("finalize", END)

        # 编译图
        self.graph = graph.compile(
            checkpointer=self.checkpointer,
            store=self.memory_store,
            interrupt_before=["human_review"] if self.review_enabled else None
        )
        return self.graph

    # ========== 节点函数定义 ==========
    def _get_review_reason(self, review) -> str:
        """获取审核触发原因"""
        reasons = []
        if review.confidence < self.config.human_review_threshold:
            reasons.append(f"置信度过低({review.confidence:.2f})")
        if review.hallucination_risk > 0.5:
            reasons.append(f"幻觉风险高({review.hallucination_risk:.2f})")
        if review.retrieval_relevance < 0.3:
            reasons.append(f"检索相关性低({review.retrieval_relevance:.2f})")

        return " | ".join(reasons) if reasons else "未知原因"

    def _process_query_node(self, state: RAGState) -> RAGState:
        """处理查询节点"""
        query = state["query"]

        # OCR文本合并
        if state.get("ocr_texts"):
            query += " " + " ".join(state["ocr_texts"])
            print(f"📷 OCR文本已并入查询：{query[:50]}...")

        # 检索历史上下文
        if self.memory_store:
            history_items = self.memory_store.search(
                (self.session_id, "conversations"),
                query=query,
                limit=2
            )
            state["history_context"] = "\n".join([
                item.value["text"] for item in history_items
            ])

        # 查询改写
        if state.get("chat_history"):
            rewritten = self.generator.rewrite_query(query, state["chat_history"])
            print(f"🔄 查询改写：{query} → {rewritten}")
            state["query"] = rewritten

        return state

    def _retrieve_node(self, state: RAGState) -> RAGState:
        """检索节点 - 增加相关性过滤"""
        query = state["query"]

        # 执行检索
        docs = asyncio.run(self.retriever.retrieve(query))

        # 添加历史上下文（保持原有逻辑）
        if state.get("history_context"):
            docs.insert(0, Document(
                page_content=state["history_context"],
                metadata={"source": "conversation_history", "score": 1.0}  # 历史给满分
            ))

        # ========== 新增：相关性过滤 ==========
        if not docs:
            # 完全无检索结果
            state["documents"] = []
            state["context"] = "（警告：未检索到任何相关资料）"
            state["sources"] = []
            print("⚠️ 未检索到任何文档")
            return state

        # 计算最高相关性分数
        max_relevance = 0.0
        for doc in docs:
            # 从metadata提取各种可能的分数
            score = (doc.metadata.get("rerank_score") or
                     doc.metadata.get("hybrid_score") or
                     doc.metadata.get("vector_score") or
                     doc.metadata.get("bm25_score", 0.0))
            if score and score > max_relevance:
                max_relevance = float(score)

        # 相关性过低过滤（阈值0.3，可配置）
        relevance_threshold = getattr(self.config, 'retrieval_relevance_threshold', 0.2)

        if max_relevance < relevance_threshold:
            # 低相关性：清空上下文，强制模型拒答
            state["documents"] = []  # 清空文档列表
            state["context"] = f"（警告：检索到的资料与问题相关性过低（{max_relevance:.2f}），知识库中可能无相关资料）"
            state["sources"] = []
            print(f"⚠️ 检索相关性过低（{max_relevance:.2f}），清空上下文")
        else:
            # 正常格式化上下文
            context_parts = []
            sources = []
            for i, doc in enumerate(docs):
                context_parts.append(f"[文档 {i + 1}] {doc.page_content}")
                sources.append({
                    "source": doc.metadata.get("source", "unknown"),
                    "content_preview": doc.page_content[:30] + "..."
                })

            state["documents"] = docs
            state["context"] = "\n\n".join(context_parts)
            state["sources"] = sources
            print(f"📚 检索到 {len(docs)} 个相关文档，最高相关性: {max_relevance:.2f}")

        return state

    def _generate_node(self, state: RAGState) -> RAGState:
        """生成节点"""
        answer = self.generator.generate(
            query=state["query"],
            context=state["context"],
            chat_history=state.get("chat_history", [])
        )

        state["answer"] = answer
        print(f"💬 生成答案完成")
        return state

    def _evaluate_node(self, state: RAGState) -> RAGState:
        """评估节点 - 钢铁容错版"""
        try:
            # 检查答案存在性
            if not state.get("answer"):
                print("⚠️ 答案为空，使用默认评估")
                state["confidence"] = 0.0
                state["iteration"] = 0
                state["review_result"] = {}
                return state

            # 检查 documents 是否存在
            if not state.get("documents"):
                print("⚠️ 文档列表为空")
                state["documents"] = []

            # 执行评估（带独立try-except）
            try:
                review = self.evaluator.evaluate(
                    state["query"],
                    state["answer"],
                    state.get("documents", []),
                    0
                )
                confidence = review.confidence
                hallucination = review.hallucination_risk
                relevance = review.retrieval_relevance
                needs_review = review.needs_human_review

                state["review_result"] = review.__dict__

            except Exception as eval_err:
                print(f"⚠️ 评估器异常: {eval_err}")
                confidence = 0.6
                hallucination = 0.5
                relevance = 0.5
                needs_review = False
                state["review_result"] = {"error": str(eval_err)}

            # 更新状态
            state["confidence"] = confidence
            state["iteration"] = state.get("iteration", 0) + 1

            print(f"📊 评估完成: 置信度={confidence:.2f}, 幻觉={hallucination:.2f}, 迭代={state['iteration']}")

            # 人工审核逻辑（仅在需要时）
            if needs_review and self.review_enabled:
                task_id = f"review_{uuid.uuid4().hex[:8]}"
                state["review_task_id"] = task_id
                state["review_status"] = "pending"
                state["review_trigger_reason"] = f"置信度低({confidence:.2f})或幻觉高({hallucination:.2f})"

                self.review_tasks[task_id] = {
                    "task_id": task_id,
                    "query": state["query"],
                    "original_answer": state["answer"],
                    "confidence": confidence,
                    "hallucination_risk": hallucination,
                    "retrieval_relevance": relevance,
                    "documents": state.get("documents", []),
                    "status": "pending",
                    "created_at": time.time(),
                    "trigger_reason": state["review_trigger_reason"]
                }
                print(f"⚠️ 触发人工审核: {task_id}")

        except Exception as e:
            print(f"❌ _evaluate_node 严重错误: {e}")
            import traceback
            traceback.print_exc()

            # 绝对不能崩，给默认值
            state["confidence"] = 0.5
            state["iteration"] = 1
            state["review_result"] = {"error": str(e)}

        return state

    def _human_review_node(self, state: RAGState) -> Command[Literal["refine", "finalize"]]:
        """人工审核节点（使用interrupt）"""
        task_id = state["review_task_id"]

        # ========== 核心：使用interrupt暂停执行 ==========
        decision = interrupt({
            "task_id": task_id,
            "query": state["query"],
            "original_answer": state["answer"],
            "confidence": state["confidence"],
            "trigger_reason": state["review_trigger_reason"],
            "message": "等待人工审核..."
        })

        # 恢复执行后的处理
        if decision is None:
            return Command(goto="finalize", update={"review_status": "approved"})

        if decision["action"] == "approved":
            return Command(goto="finalize", update={"review_status": "approved"})
        elif decision["action"] == "rejected":
            return Command(goto="refine", update={
                "review_status": "rejected",
                "query": state["query"] + " （请重新生成更准确的答案）"
            })
        elif decision["action"] == "modified":
            return Command(goto="finalize", update={
                "review_status": "modified",
                "human_modified_answer": decision["modified_answer"],
                "review_comment": decision.get("comment", ""),
                "reviewer": decision.get("reviewer", "anonymous")
            })

        return Command(goto="finalize", update={"review_status": "approved"})

    def _refine_node(self, state: RAGState) -> RAGState:
        """优化节点"""
        current_query = state["query"]
        if "（请提供更详细的回答）" not in current_query:
            state["query"] = current_query + " （请提供更详细的回答）"
        return state

    def _finalize_node(self, state: RAGState) -> RAGState:
        """结束节点：保存记忆"""
        # 保存高质量对话到长期记忆
        if self.memory_store and state["confidence"] > 0.5:
            self.memory_store.put(
                (self.session_id, "conversations"),
                f"turn_{int(time.time())}",
                {"text": f"Q: {state['query']}\nA: {state['answer'][:200]}"}
            )

        # 更新审核任务状态
        if state.get("review_task_id"):
            task_id = state["review_task_id"]
            if task_id in self.review_tasks:
                self.review_tasks[task_id]["status"] = state.get("review_status", "completed")
                self.review_tasks[task_id]["final_answer"] = state["answer"]
                self.review_tasks[task_id]["reviewed_at"] = time.time()
                self.review_tasks[task_id]["reviewer"] = state.get("reviewer")
                self.review_tasks[task_id]["review_comment"] = state.get("review_comment")

        return state

    # ========== 条件路由函数 ==========

    def _should_continue(self, state: RAGState) -> Literal["refine", "human_review", "finalize"]:
        """评估后的路由决策 - 优化审核触发"""
        iteration = state.get("iteration", 0)
        confidence = state.get("confidence", 0.0)
        relevance = state.get("review_result", {}).get("retrieval_relevance", 0.0)

        # 如果已经触发审核
        if state.get("review_task_id"):
            return "human_review"

        # 达到最大迭代次数
        if iteration >= self.config.max_iterations:
            return "finalize"

        # 置信度足够高且相关性不低，直接结束
        if (confidence >= self.config.confidence_threshold and
                relevance >= self.config.retrieval_relevance_threshold):
            return "finalize"

        # 如果相关性极低，即使置信度高也审核
        if relevance < self.config.retrieval_relevance_threshold:
            return "human_review"

        # 否则继续优化
        return "refine"

    def _should_continue_after_review(self, state: RAGState) -> Literal["refine", "finalize"]:
        """审核后的路由决策"""
        if state.get("human_modified_answer"):
            state["answer"] = state["human_modified_answer"]
            state["confidence"] = min(1.0, state.get("confidence", 0.0) + 0.2)
            return "finalize"

        if state.get("review_status") == "rejected":
            return "refine"

        return "finalize"

    # ========== 外部 API 接口 ==========

    def get_pending_reviews(self) -> List[Dict[str, Any]]:
        """获取待审核任务列表"""
        return [
            {
                "task_id": task["task_id"],
                "query": task["query"][:200] + "...",
                "confidence": task["confidence"],
                "hallucination_risk": task["hallucination_risk"],
                "retrieval_relevance": task["retrieval_relevance"],
                "trigger_reason": task["trigger_reason"],
                "created_at": task["created_at"]
            }
            for task in self.review_tasks.values()
            if task["status"] == "pending"
        ]

    def get_review_detail(self, task_id: str) -> Optional[Dict[str, Any]]:
        """获取审核任务详情"""
        task = self.review_tasks.get(task_id)
        if not task:
            return None

        return {
            "task_id": task["task_id"],
            "status": task["status"],
            "query": task["query"],
            "original_answer": task["original_answer"],
            "documents": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "content": doc.page_content[:300] + "..."
                }
                for doc in task["documents"]
            ],
            "metrics": {
                "confidence": task["confidence"],
                "hallucination_risk": task["hallucination_risk"],
                "retrieval_relevance": task["retrieval_relevance"]
            },
            "trigger_reason": task["trigger_reason"],
            "created_at": task["created_at"]
        }

    def submit_review(self, task_id: str, action: str,
                      modified_answer: Optional[str] = None,
                      comment: Optional[str] = None,
                      reviewer: Optional[str] = None) -> bool:
        """提交审核结果（与interrupt配合）"""
        if task_id not in self.review_tasks:
            print(f"❌ 审核任务不存在: {task_id}")
            return False

        if self.review_tasks[task_id]["status"] != "pending":
            print(f"❌ 审核任务状态不是pending: {task_id}")
            return False

        # 准备审核决策
        decision = {
            "action": action,
            "reviewer": reviewer or "anonymous",
            "comment": comment or ""
        }

        if action == "modified":
            if not modified_answer:
                print(f"❌ 修改答案不能为空: {task_id}")
                return False
            decision["modified_answer"] = modified_answer

        # 更新任务状态
        self.review_tasks[task_id].update({
            "status": action,
            "reviewer": decision["reviewer"],
            "review_comment": decision["comment"],
            "reviewed_at": time.time()
        })

        # ========== 恢复图执行 ==========
        try:
            config = {"configurable": {"thread_id": self.session_id}}

            # 更新状态并恢复执行
            self.graph.update_state(
                config,
                decision,  # 这会作为interrupt的返回值
                as_node="human_review"
            )

            # 异步恢复
            asyncio.create_task(self._resume_execution(task_id))

            print(f"✅ 审核提交成功: {task_id}")
            return True
        except Exception as e:
            print(f"❌ 恢复执行失败: {e}")
            return False

    async def _resume_execution(self, task_id: str):
        """异步恢复图执行"""
        try:
            config = {"configurable": {"thread_id": self.session_id}}
            async for chunk in self.graph.astream(None, config, stream_mode="updates"):
                if "__interrupt__" in chunk:
                    break
        except Exception as e:
            print(f"❌ 任务执行失败: {task_id} - {e}")

    async def query(self, question: str, chat_history: List[Dict[str, str]] = None,
                    files: Optional[List[Union[Path, str]]] = None) -> Dict[str, Any]:
        """执行查询（主入口）- OCR文件已在索引时处理"""
        if not self.graph:
            raise ValueError("请先调用 aindex_documents()")

        # 注意：files参数现在用于索引OCR文档，而非查询时处理
        # 如果需要查询时临时索引OCR文件，调用前需先执行 aindex_documents
        if files:
            await self.aindex_documents([], files, [])

        initial_state = RAGState(
            query=question,
            chat_history=chat_history or [],
            documents=[],
            context="",
            answer="",
            sources=[],
            confidence=0.0,
            iteration=0,
            history_context="",
            review_result=None,
            ocr_texts=None,
            images=None,
            review_task_id=None,
            review_status=None,
            human_modified_answer=None,
            review_comment=None,
            reviewer=None,
            review_trigger_reason=None
        )

        result = await self.graph.ainvoke(
            initial_state,
            {"configurable": {"thread_id": self.session_id}}
        )

        return {
            "answer": result["answer"],
            "confidence": result["confidence"],
            "iteration": result["iteration"],
            "sources": result["sources"],
            "review_task_id": result.get("review_task_id"),
            "review_status": result.get("review_status"),
            "review_trigger_reason": result.get("review_trigger_reason")
        }