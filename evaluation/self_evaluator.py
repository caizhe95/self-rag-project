from dataclasses import dataclass
from enum import Enum
from typing import List, Any, Dict
import time
from langchain_core.documents import Document


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ReviewResult:
    """多维度评估结果"""
    confidence: float  # 答案质量分（0-1）
    retrieval_relevance: float  # 检索相关性（0-1）
    answer_completeness: float  # 回答完整性（0-1）
    hallucination_risk: float  # 幻觉风险（0-1，越高越危险）
    latency_ms: int  # 响应延迟（毫秒）
    needs_human_review: bool
    human_review_status: ReviewStatus
    review_comment: str = ""


class SelfEvaluator:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def evaluate(self, query: str, answer: str, documents: List[Document], latency_ms: int) -> ReviewResult:
        """多维度评估入口"""

        # 维度1：置信度
        confidence = self._evaluate_confidence(answer)

        # 维度2：检索相关性
        retrieval_relevance = self._evaluate_retrieval_relevance(query, documents)

        # 维度3：回答完整性
        answer_completeness = self._evaluate_answer_completeness(query, answer)

        # 维度4：幻觉风险
        hallucination_risk = self._evaluate_hallucination_risk(answer, documents)

        # 综合判断是否需要人工审核
        needs_review = (
                self.config.HUMAN_REVIEW_ENABLED and
                (confidence < self.config.HUMAN_REVIEW_THRESHOLD or
                 hallucination_risk > 0.5 or  # 幻觉风险高必审
                 retrieval_relevance < 0.3)  # 检索质量差必审
        )

        return ReviewResult(
            confidence=confidence,
            retrieval_relevance=retrieval_relevance,
            answer_completeness=answer_completeness,
            hallucination_risk=hallucination_risk,
            latency_ms=latency_ms,
            needs_human_review=needs_review,
            human_review_status=ReviewStatus.PENDING if needs_review else ReviewStatus.APPROVED
        )

    def _evaluate_confidence(self, answer: str) -> float:
        """评估答案质量"""
        prompt = f"评估以下答案的质量（0-1分），考虑准确性、流畅性、有用性：\n\n{answer[:300]}"
        try:
            result = self.llm.invoke(prompt).strip()
            # 提取可能的数字
            import re
            match = re.search(r'(\d+\.?\d*)', result)
            score = float(match.group(1)) if match else 0.5
            return max(0.0, min(1.0, score))  # 归一化
        except Exception:
            return 0.5

    def _evaluate_retrieval_relevance(self, query: str, documents: List[Document]) -> float:
        """评估检索相关性"""
        if not documents:
            return 0.0

        # 从文档元数据中提取分数
        doc_scores = []
        for doc in documents[:3]:
            score = doc.metadata.get("rerank_score")
            if score is None:
                score = doc.metadata.get("similarity", 0.0)
            if isinstance(score, (int, float)):
                doc_scores.append(float(score))

        return max(doc_scores) if doc_scores else 0.0

    def _evaluate_answer_completeness(self, query: str, answer: str) -> float:
        """评估回答完整性（是否覆盖问题所有要点）"""
        prompt = f"""
        问题: {query}
        答案: {answer[:500]}

        评估答案是否完整回答了问题（0-1分）：
        - 1.0: 全面覆盖所有要点
        - 0.6-0.9: 回答了主要部分
        - <0.6: 遗漏关键信息

        只返回分数：
        """
        try:
            result = self.llm.invoke(prompt).strip()
            import re
            match = re.search(r'(\d+\.?\d*)', result)
            score = float(match.group(1)) if match else 0.6
            return max(0.0, min(1.0, score))
        except Exception:
            return 0.6

    def _evaluate_hallucination_risk(self, answer: str, documents: List[Document]) -> float:
        """评估幻觉风险"""
        if not documents:
            return 1.0  # 无文档支撑，风险最大

        # 提取事实性陈述
        claims = self._extract_claims(answer)
        if not claims:
            return 0.0  # 没有可验证陈述，风险低

        # 检查每个陈述是否在文档中有支撑
        unsupported_count = sum(1 for claim in claims if not self._is_supported_by_docs(claim, documents))

        # 风险 = 无支撑陈述 / 总陈述数
        risk = unsupported_count / len(claims)
        return min(1.0, risk * 2)  # 放大风险信号

    def _extract_claims(self, answer: str) -> List[str]:
        """用LLM提取事实性陈述"""
        if len(answer) < 20:  # 太短，无需提取
            return []

        prompt = f"提取以下文本中的可验证事实陈述（每行一个，只提取明确的、可验证的事实）：\n\n{answer[:400]}"
        try:
            result = self.llm.invoke(prompt)
            # 过滤空行和无效行
            claims = [line.strip() for line in result.split("\n")
                     if line.strip() and len(line.strip()) > 5]
            return claims[:10]  # 最多10个，避免过多
        except Exception:
            return []

    def _is_supported_by_docs(self, claim: str, documents: List[Document]) -> bool:
        """检查陈述是否在文档中有支撑"""
        if not documents or not claim:
            return False

        claim_lower = claim.lower()
        # 简单匹配：文档中是否包含该陈述
        for doc in documents:
            if claim_lower in doc.page_content.lower():
                return True

        # 语义匹配：使用LLM判断蕴含关系
        try:
            doc_content = documents[0].page_content[:300]
            prompt = f"根据文档判断陈述是否被支持（回答'是'或'否'）：\n\n文档: {doc_content}\n\n陈述: {claim}"
            result = self.llm.invoke(prompt).lower()
            return "是" in result or "yes" in result or "true" in result
        except Exception:
            return False

    def request_human_review(self, query: str, answer: str) -> dict:
        """生成人工审核任务"""
        return {
            "task_id": f"review_{hash(query + answer) % 100000}",
            "query": query,
            "answer": answer,
            "status": "pending",
            "timestamp": None,
            "reviewer": None,
            "comment": ""
        }