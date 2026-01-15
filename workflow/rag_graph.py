import asyncio
import time
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.types import interrupt, Command
from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    documents: List[Document]
    answer: str
    sources: List[Dict]
    confidence: float
    iteration: int
    review_result: Any
    human_review_decision: bool
    human_review_comment: str
    start_time: float


class RAGWorkflow:
    def __init__(self, retriever, evaluator, llm, config):
        self.retriever = retriever
        self.evaluator = evaluator
        self.llm = llm
        self.config = config
        self.graph = self._build()

    def _build(self):
        workflow = StateGraph(RAGState)

        async def retrieve(state: RAGState):
            """检索"""
            docs = await self.retriever.aretrieve_with_cache(state["query"])

            sources = [{"source": doc.metadata.get("source", "unknown"),
                        "method": doc.metadata.get("retrieval_method")}
                       for doc in docs]
            return {"documents": docs, "sources": sources}

        async def generate(state: RAGState):
            """生成"""
            context = "\n\n".join([doc.page_content for doc in state["documents"]])
            prompt = f"基于上下文回答:\n\n{context}\n\n问题: {state['query']}"

            answer = await asyncio.to_thread(self.llm.invoke, prompt)
            return {"answer": answer}

        async def evaluate(state: RAGState):
            """评估"""
            # 计算延迟（毫秒）
            latency_ms = int((time.time() - state.get("start_time", time.time())) * 1000)

            # 多维度评估（传入延迟参数）
            review = await asyncio.to_thread(
                self.evaluator.evaluate,
                state["query"],
                state["answer"],
                state["documents"],
                latency_ms
            )

            if review.needs_human_review:
                review_task = self.evaluator.request_human_review(state["query"], state["answer"])

                user_input = await asyncio.to_thread(interrupt, review_task)

                return Command(
                    update={
                        "human_review_decision": user_input.get("approved", False),
                        "human_review_comment": user_input.get("comment", ""),
                        "confidence": max(review.confidence, user_input.get("updated_confidence", 0.7))
                    }
                )

            return {"review_result": review.__dict__, "confidence": review.confidence}

        async def decide_next(state: RAGState):
            """决策"""
            if state.get("human_review_decision") is True:
                return "finalize"
            if state["confidence"] >= self.config.CONFIDENCE_THRESHOLD:
                return "finalize"
            if state["iteration"] < self.config.MAX_ITERATIONS:
                return "refine"
            return "finalize"

        async def refine(state: RAGState):
            """优化查询"""
            return {
                "query": state["query"] + " （更详细的）",
                "iteration": state["iteration"] + 1
            }

        async def finalize(state: RAGState):
            """生成最终答案"""
            if state["confidence"] < self.config.CONFIDENCE_THRESHOLD:
                final_answer = f"{state['answer']}\n\n 置信度较低（{state['confidence']:.2f}），建议核实。"
            else:
                final_answer = state["answer"]
            return {"answer": final_answer}

        # 注册异步节点
        workflow.add_node("retrieve", retrieve)
        workflow.add_node("generate", generate)
        workflow.add_node("evaluate", evaluate)
        workflow.add_node("refine", refine)
        workflow.add_node("finalize", finalize)

        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", "evaluate")
        workflow.add_conditional_edges("evaluate", decide_next, {
            "refine": "refine",
            "finalize": "finalize"
        })
        workflow.add_edge("refine", "retrieve")
        workflow.add_edge("finalize", END)

        return workflow.compile()