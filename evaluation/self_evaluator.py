# core/self_evaluator.py - 策略模式自适应评估器（实习面试版）
from dataclasses import dataclass
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import re
import time

from tests.test_monitor import monitor_evaluation


@dataclass
class ReviewResult:
    """评估结果"""
    confidence: float
    retrieval_relevance: float
    answer_completeness: float
    hallucination_risk: float
    latency_ms: int
    needs_human_review: bool
    review_comment: str = ""


class SelfEvaluator:
    """Self-RAG 评估器 - 自动适配小模型(规则)和大模型(LLM验证)"""

    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

        # 从策略配置获取参数（自动适配小/大模型）
        self.use_llm_contradiction = getattr(config, 'use_llm_contradiction', False)
        self.extract_claims_max = getattr(config, 'extract_claims_max', 3)
        self.strict_mode = getattr(config, 'strict_mode', False)
        self.human_review_threshold = getattr(config, 'human_review_threshold', 0.4)

        # 打印当前策略（一目了然）
        model_type = "大模型(LLM矛盾检测)" if self.use_llm_contradiction else "小模型(规则矛盾检测)"
        print(f"📊 评估器初始化: {model_type}, 严格模式={self.strict_mode}")

        # 初始化提示词
        self._init_prompts()

    def _init_prompts(self):
        """初始化提示词（小模型极简，大模型详细）"""
        if self.strict_mode:
            # 大模型：详细提示词
            self.confidence_prompt = PromptTemplate.from_template(
                """评估回答质量（0-1分），严格区分事实与推测：

                0.9-1.0：完全基于资料，无外部推测
                0.7-0.8：基于资料，含必要术语解释
                0.5-0.6：部分基于资料，部分合理推断
                0.3-0.4：大多推测，与资料关联弱
                0.0-0.2：与资料矛盾或无法验证

                只返回0-1之间的数字，小数点后1位。

                资料：{context}
                问题：{query}
                回答：{answer}
                评分："""
            )
        else:
            # 小模型：极简提示词（防超时）
            self.confidence_prompt = PromptTemplate.from_template(
                "评估质量(0-1)，只返回数字。资料：{context} 问题：{query} 回答：{answer} 分数："
            )

        self.confidence_chain = self.confidence_prompt | self.llm | StrOutputParser()

    @monitor_evaluation
    def evaluate(self, query: str, answer: str, documents: List[Document], latency_ms: int) -> ReviewResult:
        """
        主评估入口 - 内部容错，对外始终返回合法结果
        自动适配：小模型快速评估，大模型精准评估
        """

        def safe_eval(func, default, *args, **kwargs):
            """安全执行评估函数，异常时返回默认值"""
            try:
                return func(*args, **kwargs)
            except Exception as e:
                print(f"⚠️ 评估子项失败({func.__name__}): {e}")
                return default

        start_time = time.time()

        # 并行计算各维度（带容错）
        confidence = safe_eval(self._evaluate_confidence, 0.6, answer)
        retrieval_relevance = safe_eval(self._evaluate_retrieval_relevance, 0.5, query, documents)
        answer_completeness = safe_eval(self._evaluate_completeness, 0.6, query, answer)
        hallucination_risk = safe_eval(self._evaluate_hallucination_risk, 0.5, answer, documents)

        eval_latency = int((time.time() - start_time) * 1000)

        # 组合策略触发审核（避免单一指标误杀）
        needs_review = False
        if getattr(self.config, 'human_review_enabled', False):
            # 策略：低置信度 + 高幻觉 同时满足，或检索完全失败
            if (confidence < self.human_review_threshold and hallucination_risk > 0.6):
                needs_review = True
            elif retrieval_relevance < 0.2:
                needs_review = True

        return ReviewResult(
            confidence=confidence,
            retrieval_relevance=retrieval_relevance,
            answer_completeness=answer_completeness,
            hallucination_risk=hallucination_risk,
            latency_ms=latency_ms + eval_latency,
            needs_human_review=needs_review,
            review_comment="评估完成"
        )

    def _evaluate_confidence(self, answer: str) -> float:
        """置信度评估 - 解析数字并归一化"""
        result = self.confidence_chain.invoke({
            "answer": answer[:500],
            "context": "",
            "query": ""
        }).strip()

        # 提取数字（适配各种格式）
        m = re.search(r"(0?\.\d+|1\.0|1)", result)
        if m:
            score = float(m.group(1))
            # 智能归一化：如果是8-10分制，转为0-1
            if score > 1.0 and score <= 10:
                score = score / 10
            return min(max(score, 0.0), 1.0)
        return 0.6

    def _evaluate_retrieval_relevance(self, query: str, documents: List[Document]) -> float:
        """检索相关性 - 基于文档分数"""
        if not documents:
            return 0.0

        doc_scores = []
        for doc in documents[:3]:
            score = (doc.metadata.get("rerank_score") or
                     doc.metadata.get("hybrid_score") or
                     doc.metadata.get("vector_score") or
                     doc.metadata.get("bm25_score") or
                     doc.metadata.get("score", 0.0))

            if score is not None:
                doc_scores.append(float(score))

        # 没有分数但有文档，给中等分（小模型宽容策略）
        if not doc_scores and documents:
            return 0.6 if not self.strict_mode else 0.4

        return max(doc_scores) if doc_scores else 0.0

    def _evaluate_completeness(self, query: str, answer: str) -> float:
        """回答完整性 - 小模型用启发式，大模型用LLM判断"""

        # 小模型策略：简单启发式（不调用LLM，省资源）
        if not self.strict_mode:
            length = len(answer)
            if 50 <= length <= 200:
                return 0.8
            elif length > 20:
                return 0.6
            else:
                return 0.3

        # 大模型策略：LLM判断（精准但耗资源）
        prompt = f"""问题：{query}
        回答：{answer[:300]}

        评估回答完整性（0-1）：
        - 1.0：全面覆盖所有要点
        - 0.6-0.9：回答了主要部分  
        - <0.6：遗漏关键信息

        只返回数字："""

        try:
            result = self.llm.invoke(prompt).strip()
            match = re.search(r'(\d+\.?\d*)', result)
            score = float(match.group(1)) if match else 0.6
            return min(max(score, 0.0), 1.0)
        except:
            return 0.6

    def _evaluate_hallucination_risk(self, answer: str, documents: List[Document]) -> float:
        """
        幻觉风险评估 - 策略模式核心
        小模型：规则检测（快速）
        大模型：LLM验证（精准）
        """
        if not documents:
            return 1.0

        claims = self._extract_claims(answer)
        if not claims:
            return 0.0

        # 检测未支撑的陈述
        unsupported = 0
        for claim in claims:
            # 策略选择：大模型用LLM验证，小模型用规则
            is_supported = self._is_supported_by_docs(claim, documents)

            if not is_supported:
                # 小模型宽容：短句(<15字)不视为幻觉（可能是常识）
                if not self.strict_mode and len(claim) < 15:
                    continue
                unsupported += 1

        if not claims:
            return 0.0

        # 计算风险比例（小模型封顶0.8避免过度惩罚）
        risk_ratio = unsupported / len(claims)
        max_risk = 0.8 if not self.strict_mode else 1.0

        return min(risk_ratio * 1.2, max_risk)

    def _extract_claims(self, answer: str) -> List[str]:
        """
        提取事实陈述 - 策略化
        小模型：简单分割（快速，不耗token）
        大模型：LLM提取（精准）
        """
        if len(answer) < 20:
            return []

        max_claims = self.extract_claims_max

        # 小模型策略：简单按句号分割（不调用LLM）
        if not self.strict_mode:
            import re
            # 保护小数点
            text = re.sub(r'(\d)\.(\d)', r'\1[DOT]\2', answer)
            sentences = re.split(r'[。！？\n]+', text)

            claims = []
            for s in sentences:
                s = s.strip().replace('[DOT]', '.')
                if len(s) > 5 and len(s) < 100:
                    claims.append(s)
            return claims[:max_claims]

        # 大模型策略：LLM智能提取
        prompt = f"""从以下文本中提取{max_claims}个独立的事实陈述（每行一个）：
        要求：明确的、可验证的短句，不要总结性语句

        文本：{answer[:400]}

        事实陈述："""

        try:
            result = self.llm.invoke(prompt).strip()
            claims = [
                line.strip() for line in result.split("\n")
                if line.strip() and len(line) > 5 and not line.startswith("•")
            ]
            return claims[:max_claims]
        except Exception as e:
            print(f"⚠️ LLM提取claims失败: {e}，退回到规则提取")
            # 失败时退回到简单分割
            return [s.strip() for s in answer.split("。") if len(s.strip()) > 10][:max_claims]

    def _is_supported_by_docs(self, claim: str, documents: List[Document]) -> bool:
        """
        检查陈述是否有文档支撑 - 策略模式统一入口
        根据配置自动选择规则或LLM验证
        """
        if not claim or not documents:
            return False

        # 策略分支：大模型用LLM验证，小模型用规则
        if self.use_llm_contradiction:
            # 大模型：精准LLM验证
            return self._llm_contradiction_check(claim, documents)
        else:
            # 小模型：轻量级规则验证
            return self._rule_contradiction_check(claim, documents)

    def _rule_contradiction_check(self, claim: str, documents: List[Document]) -> bool:
        """轻量级规则检测（小模型用）- 基于关键词匹配"""
        claim_lower = claim.lower()

        # 1. 简单子串匹配（快速）
        for doc in documents:
            if claim_lower in doc.page_content.lower():
                return True

        # 2. 关键词匹配（60%以上关键词出现即认为支持）
        claim_words = set(claim_lower.split())
        if len(claim_words) < 3:
            return False  # 太短无法判断

        for doc in documents:
            doc_text = doc.page_content.lower()
            doc_words = set(doc_text.split())
            overlap = len(claim_words & doc_words) / len(claim_words)
            if overlap > 0.6:
                return True

        return False

    def _llm_contradiction_check(self, claim: str, documents: List[Document]) -> bool:
        """
        LLM-based矛盾检测（大模型用）- 精准但耗时
        适用于 deepseek-33b/qwen-14b 等大模型
        """
        # 取最相关的1-2篇文档（节省token）
        sorted_docs = sorted(
            documents,
            key=lambda d: d.metadata.get("rerank_score", 0) or d.metadata.get("vector_score", 0),
            reverse=True
        )[:2]

        # 截断文档内容（防止超出上下文）
        context_parts = []
        for i, doc in enumerate(sorted_docs, 1):
            content = doc.page_content[:300].replace("\n", " ")
            context_parts.append(f"[文档{i}] {content}...")

        context = "\n".join(context_parts)

        # 明确的三分类判断prompt
        prompt = f"""作为事实核查专家，判断以下陈述是否与提供的文档内容矛盾。

【文档内容】
{context}

【待核查陈述】
"{claim}"

【任务定义】
- "矛盾"：文档明确说了与陈述相反的内容（如文档说"支持A"，陈述说"不支持A"）
- "不矛盾"：文档支持该陈述，或文档未提及该陈述（无法验证不算矛盾）
- 注意：文档未提及的内容不要判为矛盾，应判为"不矛盾"

【输出要求】
只回复以下两个词之一，不要解释：
矛盾 / 不矛盾

判断结果："""

        try:
            # 调用大模型判断
            result = self.llm.invoke(prompt).strip().lower()

            # 解析结果（包含"矛盾"且不含"不矛盾"）
            is_contradictory = ("矛盾" in result or "contradict" in result) and "不矛盾" not in result

            if is_contradictory:
                print(f"  ⚠️ LLM检测到矛盾: '{claim[:30]}...'")

            # 如果矛盾，返回False（表示"不支持"）
            # 如果不矛盾，返回True（表示"支持"或"无法验证但不矛盾"）
            return not is_contradictory

        except Exception as e:
            print(f"⚠️ LLM矛盾检测失败: {e}，退回到规则检测")
            # 失败时退回到规则检测，保证不阻断流程
            return self._rule_contradiction_check(claim, documents)