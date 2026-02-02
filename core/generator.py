# core/generator.py
from typing import List, Dict, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM
from config.setting import RAGConfig


class Generator:
    """生成器：基于上下文生成回答（面试亮点：展示LangChain链式调用 + Self-RAG评估）"""

    def __init__(self, ollama_base_url: str, llm_model: str, temperature: float = 0.1):
        """
        初始化生成器

        Args:
            ollama_base_url: Ollama服务地址
            llm_model: 模型名称
            temperature: 生成温度
        """
        # ========== 面试重点：在内部初始化LLM，而不是外部传入 ==========
        self.llm = OllamaLLM(
            model=llm_model,
            base_url=ollama_base_url,
            temperature=temperature
        )

        # ========== RAG 生成提示词（严谨版） ==========
        self.rag_prompt = ChatPromptTemplate.from_messages([
            ("system", """【角色】企业知识库助手，基于参考资料回答。

        【第一部分：实际资料 - 必须基于这些内容回答】
        资料来源标记为 [文档 1]、[文档 2]：
        {context}

        【第二部分：输出格式 - 严格遵守】
        1. 正文要求：
           - 基于【第一部分】回答，不要参考【第三部分】的内容
           - 引用标记必须用 [1] 和 [2] 和 [3] 等以此类推，严禁使用 [n]
           - 事实陈述后紧跟编号，如：LangChain是框架[1]

        2. 来源汇总（必须单独一行）：
           - 以 docs: 开头
           - 格式：[1] 实际文档名.txt: 前30字摘要
           - 多来源用 | 分隔

        【第三部分：格式示例 - 仅参考结构，内容虚假】
        示例正文：这是一个事实[1]，这是另一个事实[2]。
        示例来源：docs: [1] xxx.txt: 文档一的内容摘要... | [2] yyy.txt: 文档二的内容摘要...

        【严禁事项】
        × 禁止照抄示例中的 xxx.txt、yyy.txt 等字样
        × 禁止使用 [n]，必须用 [1] 或 [2]等
        × 禁止把示例中的虚构内容当成事实

        【自检清单】
        输出前确认：
        □ 正文引用的是 [1] 和 [2]等，不是 [n]
        □ docs行使用的是实际文档名，不是xxx/yyy等
        □ 内容来自【第一部分】，不是【第三部分】"""),
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            ("human", "{query}")
        ])

        # ========== 查询改写提示词（精准版） ==========
        self.rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", """优化用户查询以提升检索效果。

        【策略】
        - 含指代词（其/它/该）：补全主体，如"它的特点？"→"该技术的特点？"
        - 完整问句：原样返回，禁止过度改写
        - 关键词缺失：补充同义词，如"LLM"→"大语言模型(LLM)"

        【约束】
        - 保持原意，禁止扩写
        - 直接输出改写后的问题，无需解释"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "对话历史：{chat_history}\n当前问题：{query}\n优化后问题：")
        ])

        # ========== 置信度评估提示词（幻觉感知版） ==========
        self.confidence_prompt = ChatPromptTemplate.from_messages([
            ("system", """评估回答质量（0-1分），区分事实与推测：

        【评分细则】
        0.9-1.0：完全基于资料，无外部推测
        0.7-0.8：基于资料，含必要术语解释（如"API即接口"）
        0.5-0.6：部分基于资料，部分合理推断
        0.3-0.4：大多推测，与资料关联弱
        0.0-0.2：与资料矛盾或无法验证

        【关键原则】
        - 术语解释≠幻觉，不扣分
        - 资料未提及的具体数据=幻觉，扣分

        只返回0-1之间的数字，小数点后保留1位。"""),
            ("human", """参考资料：
        {context}

        问题：{query}
        回答：{answer}

        质量评分：""")
        ])

        # ========== 面试重点：显式构建链，展示管道操作符 ==========
        self.rewrite_chain = self.rewrite_prompt | self.llm | StrOutputParser()
        self.rag_chain = self.rag_prompt | self.llm | StrOutputParser()
        self.confidence_chain = self.confidence_prompt | self.llm | StrOutputParser()

    def rewrite_query(self, query: str, chat_history: List[Dict[str, str]]) -> str:
        """
        根据对话历史改写问题

        Args:
            query: 用户原始问题
            chat_history: 对话历史

        Returns:
            改写后的查询
        """
        if not chat_history:
            return query

        # 格式化历史
        messages = self._format_history(chat_history[-4:])  # 只用最近4轮

        # ========== 使用链式调用 ==========
        return self.rewrite_chain.invoke({
            "query": query,
            "chat_history": messages
        }).strip()

    def generate(self, query: str, context: str, chat_history: List[Dict[str, str]] = None) -> str:
        """
        生成回答

        Args:
            query: 用户问题
            context: 检索到的上下文
            chat_history: 对话历史

        Returns:
            生成的答案
        """
        messages = []
        if chat_history:
            messages = self._format_history(chat_history[-4:])

        # ========== 使用链式调用 ==========
        return self.rag_chain.invoke({
            "query": query,
            "context": context,
            "chat_history": messages
        }).strip()

    def evaluate_confidence(self, query: str, context: str, answer: str) -> float:
        """
        评估答案置信度（Self-RAG核心功能）

        Args:
            query: 用户问题
            context: 检索上下文
            answer: 生成的答案

        Returns:
            0-1之间的置信度分数
        """
        # ========== 使用链式调用 ==========
        result = self.confidence_chain.invoke({
            "query": query,
            "context": context,
            "answer": answer
        })

        try:
            # 提取数字并归一化
            import re
            match = re.search(r'(\d+\.?\d*)', result)
            score = float(match.group(1)) if match else 0.5
            return min(max(score, 0.0), 1.0)
        except Exception:
            return 0.5

    def _format_history(self, history: List[Dict[str, str]]) -> List[BaseMessage]:
        """
        格式化历史消息为 LangChain 格式

        Args:
            history: 对话历史字典列表

        Returns:
            BaseMessage 对象列表
        """
        messages = []
        for msg in history:
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))
        return messages