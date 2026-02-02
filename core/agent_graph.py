# core/agent_graph.py
from typing import Annotated, Sequence, TypedDict, Literal, Optional
import operator

from langchain_core.messages import BaseMessage, SystemMessage, HumanMessage, AIMessage, ToolMessage
from langchain_ollama import OllamaLLM
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from langgraph.checkpoint.memory import MemorySaver

from core.agent_tools import (
    retrieve_knowledge,
    evaluate_answer_quality,
    process_document,
    trigger_human_review,
    check_system_status,
    init_agent_context
)
from core.rag_chain import SelfRAGChain
from config.setting import RAGConfig


class AgentState(TypedDict):
    """Agent状态定义"""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    iteration_count: int
    confidence_score: float
    review_task_id: Optional[str]
    last_query: str
    needs_review: bool


class SelfRAGAgent:
    """
    Self-RAG Agent封装（适配你现有项目）
    使用显式Schema定义工具，符合LangChain 1.0+标准
    """

    def __init__(self, rag_chain: SelfRAGChain, config: RAGConfig):
        self.rag_chain = rag_chain
        self.config = config

        # 初始化工具上下文
        init_agent_context(rag_chain, config)

        # 工具列表
        self.tools = [
            retrieve_knowledge,
            evaluate_answer_quality,
            process_document,
            trigger_human_review,
            check_system_status
        ]

        # 绑定工具的LLM
        self.llm = OllamaLLM(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature
        )
        self.llm_with_tools = self.llm.bind_tools(self.tools)

        # 预构建工具节点（自动处理Schema验证和State注入）
        self.tool_node = ToolNode(self.tools)

        # 构建图
        self.graph = self._build_graph()

        # 编译（启用interrupt）
        self.compiled = self.graph.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_review_node"]  # 人工审核前暂停
        )

    def _build_graph(self) -> StateGraph:
        """构建Agent工作流"""

        async def agent_node(state: AgentState) -> Command[
            Literal["tools", "evaluate", "human_review_node", "__end__"]]:
            """
            Agent决策节点：决定调用工具还是生成答案
            """
            messages = state["messages"]

            # 系统提示（指导Agent使用工具）
            system_msg = SystemMessage(content="""你是Self-RAG智能助手，必须使用工具获取信息。

工作流程：
1. 首先调用 retrieve_knowledge 检索资料（必须）
2. 基于检索结果回答用户问题
3. 调用 evaluate_answer_quality 评估答案质量
4. 如果评估显示置信度低(<0.5)或幻觉风险高，调用 trigger_human_review

约束：
- 禁止编造知识库中没有的信息
- 回答必须基于检索到的文档
- 每次回答后必须进行质量评估""")

            response = await self.llm_with_tools.ainvoke([system_msg] + list(messages))

            # 检查是否触发工具调用
            if hasattr(response, 'tool_calls') and response.tool_calls:
                tool_name = response.tool_calls[0]["name"]

                # 如果是人工审核工具，直接路由到审核节点
                if tool_name == "trigger_human_review":
                    return Command(
                        goto="human_review_node",
                        update={"messages": [response], "needs_review": True}
                    )

                return Command(goto="tools", update={"messages": [response]})

            # 没有工具调用，进入评估节点
            return Command(
                goto="evaluate",
                update={"messages": [response], "last_query": messages[-1].content if messages else ""}
            )

        async def evaluate_node(state: AgentState) -> Command[Literal["agent", "human_review_node", "__end__"]]:
            """
            评估节点：检查答案质量，决定是否结束或继续优化
            """
            # 提取最后生成的答案
            last_answer = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and not msg.tool_calls:
                    last_answer = msg.content
                    break

            if not last_answer:
                return Command(goto=END)

            # 提取检索到的上下文（从之前的ToolMessage）
            contexts = []
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage) and msg.name == "retrieve_knowledge":
                    # 解析artifact（存储在tool_message.artifact中）
                    if hasattr(msg, 'artifact') and msg.artifact:
                        contexts = [doc.page_content for doc in msg.artifact[:3]]
                    break

            # 获取用户问题
            user_query = state.get("last_query", "")

            # 调用评估工具
            eval_result = evaluate_answer_quality.invoke({
                "query": user_query,
                "answer": last_answer,
                "contexts": contexts,
                "state": state  # 自动注入
            })

            # 解析置信度（从文本中提取）
            import re
            match = re.search(r'置信度:\s*(\d+)%', eval_result)
            confidence = int(match.group(1)) / 100 if match else 0.5

            # 更新状态
            new_state = {
                "confidence_score": confidence,
                "iteration_count": state["iteration_count"] + 1,
                "messages": state["messages"] + [AIMessage(content=f"[系统评估] {eval_result}")]
            }

            # 决策逻辑
            if confidence >= self.config.confidence_threshold:
                return Command(goto=END, update=new_state)

            if self.config.human_review_enabled and confidence < self.config.human_review_threshold:
                new_state["needs_review"] = True
                return Command(goto="human_review_node", update=new_state)

            # 继续迭代优化
            if new_state["iteration_count"] < self.config.max_iterations:
                return Command(
                    goto="agent",
                    update={
                        **new_state,
                        "messages": new_state["messages"] + [HumanMessage(content="请基于资料重新生成更准确的回答")]
                    }
                )

            return Command(goto=END, update=new_state)

        def human_review_node(state: AgentState) -> Command[Literal["agent", "__end__"]]:
            """
            人工审核节点：处理interrupt恢复
            """
            if not state.get("needs_review"):
                return Command(goto=END)

            # 等待外部审核结果
            decision = interrupt({
                "type": "human_review",
                "task_id": state.get("review_task_id"),
                "query": state.get("last_query"),
                "answer": next(
                    (m.content for m in reversed(state["messages"])
                     if isinstance(m, AIMessage) and not m.tool_calls),
                    ""
                ),
                "confidence": state.get("confidence_score")
            })

            # 处理审核结果
            action = decision.get("action")

            if action == "approved":
                return Command(goto=END, update={"needs_review": False})

            elif action == "rejected":
                return Command(
                    goto="agent",
                    update={
                        "needs_review": False,
                        "messages": state["messages"] + [HumanMessage(content="审核拒绝，请重新生成更准确、更详细的回答")]
                    }
                )

            elif action == "modified":
                modified = decision.get("modified_answer", "")
                return Command(
                    goto=END,
                    update={
                        "messages": state["messages"] + [AIMessage(content=modified)],
                        "needs_review": False
                    }
                )

            return Command(goto=END, update={"needs_review": False})

        # 构建图
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", self.tool_node)  # ToolNode自动处理Schema验证
        workflow.add_node("evaluate", evaluate_node)
        workflow.add_node("human_review_node", human_review_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent")

        return workflow.compile()

    async def query(self, question: str, session_id: str = "default") -> dict:
        """对外查询接口"""
        initial_state = AgentState(
            messages=[HumanMessage(content=question)],
            iteration_count=0,
            confidence_score=0.0,
            review_task_id=None,
            last_query=question,
            needs_review=False
        )

        config = {"configurable": {"thread_id": session_id}}
        result = await self.compiled.ainvoke(initial_state, config)

        # 提取最终答案（过滤掉系统评估消息）
        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not msg.tool_calls:
                if not msg.content.startswith("[系统评估]"):
                    final_answer = msg.content
                    break

        return {
            "answer": final_answer,
            "confidence": result.get("confidence_score", 0),
            "iterations": result.get("iteration_count", 0),
            "needs_review": result.get("needs_review", False),
            "review_task_id": result.get("review_task_id")
        }

    async def submit_review(self, task_id: str, action: str, modified_answer: str = None):
        """提交人工审核结果（恢复执行）"""
        from langgraph.types import Command

        resume_cmd = Command(resume={
            "action": action,
            "modified_answer": modified_answer
        })

        return await self.compiled.ainvoke(
            resume_cmd,
            {"configurable": {"thread_id": "default"}}
        )