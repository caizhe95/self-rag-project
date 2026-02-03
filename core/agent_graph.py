# core/agent_graph.py
from typing import Annotated, Sequence, TypedDict, Literal, Optional, Any
import operator
import asyncio
import re

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
    Self-RAG Agent封装（不使用 bind_tools，兼容 OllamaLLM）
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

        # 基础LLM（不使用 bind_tools）
        self.llm = OllamaLLM(
            model=config.llm_model,
            base_url=config.ollama_base_url,
            temperature=config.temperature
        )
        # 不使用 self.llm_with_tools，直接调用工具

        # 预构建工具节点
        self.tool_node = ToolNode(self.tools)

        # 构建图
        self.graph = self._build_graph()

        # 编译（启用interrupt）
        self.compiled = self.graph

    def _build_graph(self) -> Any:
        """构建Agent工作流"""

        async def agent_node(state: AgentState) -> Command[
            Literal["tools", "evaluate", "human_review_node", "__end__"]]:
            """
            Agent决策节点：决定调用工具还是生成答案
            不使用 bind_tools，通过 prompt 指导工具调用
            """
            messages = state["messages"]

            # 系统提示（指导Agent使用工具）
            system_msg = SystemMessage(content="""你是Self-RAG智能助手。

可用工具：
1. retrieve_knowledge - 从知识库检索资料（必需的第一步）
2. evaluate_answer_quality - 评估答案质量
3. trigger_human_review - 触发人工审核（当置信度<0.5时使用）

工作流程：
1. 首先调用 retrieve_knowledge 检索资料
2. 基于检索结果回答用户问题
3. 调用 evaluate_answer_quality 评估答案质量
4. 如果评估显示置信度低(<0.5)，调用 trigger_human_review

请通过以下格式调用工具：
TOOL: tool_name
参数: {"key": "value"}

或直接回答问题（当你已有足够信息时）。""")

            # 直接调用LLM，不绑定工具
            response = await self.llm.ainvoke([system_msg] + list(messages))

            # 解析工具调用（从文本中解析）
            content = response.content if hasattr(response, 'content') else str(response)

            # 检查是否包含工具调用标记
            if "TOOL:" in content:
                lines = content.strip().split('\n')
                tool_line = next((l for l in lines if l.startswith("TOOL:")), None)

                if tool_line:
                    tool_name = tool_line.replace("TOOL:", "").strip().split()[0]

                    # 提取参数
                    param_line = next((l for l in lines if l.startswith("参数:")), None)
                    params = {}
                    if param_line:
                        import json
                        try:
                            params_str = param_line.replace("参数:", "").strip()
                            params = json.loads(params_str)
                        except:
                            pass

                    # 创建工具调用消息
                    tool_msg = AIMessage(
                        content="",
                        tool_calls=[{
                            "name": tool_name,
                            "args": params,
                            "id": f"call_{hash(tool_name)}"
                        }]
                    )

                    # 如果是人工审核工具
                    if tool_name == "trigger_human_review":
                        return Command(
                            goto="human_review_node",
                            update={"messages": list(messages) + [tool_msg], "needs_review": True}
                        )

                    return Command(goto="tools", update={"messages": list(messages) + [tool_msg]})

            # 没有工具调用，进入评估节点
            return Command(
                goto="evaluate",
                update={
                    "messages": list(messages) + [response] if isinstance(response, BaseMessage) else list(messages) + [
                        AIMessage(content=content)],
                    "last_query": messages[-1].content if messages else ""
                }
            )

        async def evaluate_node(state: AgentState) -> Command[Literal["agent", "human_review_node", "__end__"]]:
            """
            评估节点：检查答案质量
            """
            # 提取最后生成的答案
            last_answer = ""
            for msg in reversed(state["messages"]):
                if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
                    if not msg.content.startswith("[系统评估]") and msg.content.strip():
                        last_answer = msg.content
                        break

            if not last_answer:
                return Command(goto=END)

            # 提取检索到的上下文
            contexts = []
            for msg in reversed(state["messages"]):
                if isinstance(msg, ToolMessage) and getattr(msg, 'name', None) == "retrieve_knowledge":
                    contexts = [msg.content] if msg.content else []
                    break

            # 获取用户问题
            user_query = state.get("last_query", "")

            # 直接调用评估工具（不通过LLM绑定）
            try:
                eval_result = evaluate_answer_quality.invoke({
                    "query": user_query,
                    "answer": last_answer,
                    "contexts": contexts
                })
            except Exception as e:
                eval_result = f"评估失败: {str(e)}，默认置信度50%"

            # 解析置信度
            match = re.search(r'置信度:\s*(\d+)%', eval_result)
            confidence = int(match.group(1)) / 100 if match else 0.5

            # 更新状态
            new_state = {
                "confidence_score": confidence,
                "iteration_count": state["iteration_count"] + 1,
                "messages": list(state["messages"]) + [AIMessage(content=f"[系统评估] {eval_result}")]
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
                        "messages": list(new_state["messages"]) + [
                            HumanMessage(content="请基于资料重新生成更准确的回答")]
                    }
                )

            return Command(goto=END, update=new_state)

        def human_review_node(state: AgentState) -> Command[Literal["agent", "__end__"]]:
            """
            人工审核节点
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
                     if isinstance(m, AIMessage) and not getattr(m, 'tool_calls', None)),
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
                        "messages": list(state["messages"]) + [
                            HumanMessage(content="审核拒绝，请重新生成更准确、更详细的回答")]
                    }
                )

            elif action == "modified":
                modified = decision.get("modified_answer", "")
                return Command(
                    goto=END,
                    update={
                        "messages": list(state["messages"]) + [AIMessage(content=modified)],
                        "needs_review": False
                    }
                )

            return Command(goto=END, update={"needs_review": False})

        # 构建图
        workflow = StateGraph(AgentState)

        workflow.add_node("agent", agent_node)
        workflow.add_node("tools", self.tool_node)
        workflow.add_node("evaluate", evaluate_node)
        workflow.add_node("human_review_node", human_review_node)

        workflow.add_edge(START, "agent")
        workflow.add_edge("tools", "agent")

        return workflow.compile(
            checkpointer=MemorySaver(),
            interrupt_before=["human_review_node"]
        )

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

        # 提取最终答案
        final_answer = ""
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage) and not getattr(msg, 'tool_calls', None):
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
        """提交人工审核结果"""
        from langgraph.types import Command

        resume_cmd = Command(resume={
            "action": action,
            "modified_answer": modified_answer
        })

        return await self.compiled.ainvoke(
            resume_cmd,
            {"configurable": {"thread_id": "default"}}
        )