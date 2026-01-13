from dataclasses import dataclass
from enum import Enum
from typing import List, Any


class ReviewStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


@dataclass
class ReviewResult:
    confidence: float
    needs_human_review: bool
    human_review_status: ReviewStatus
    review_comment: str = ""


class SelfEvaluator:
    def __init__(self, llm, config):
        self.llm = llm
        self.config = config

    def evaluate(self, answer: str, contexts: List) -> ReviewResult:
        prompt = f"评估答案质量（0-1分）：{answer[:200]}"
        try:
            score = float(self.llm.invoke(prompt).strip())
        except:
            score = 0.5

        needs_review = (
                self.config.HUMAN_REVIEW_ENABLED and
                score < self.config.HUMAN_REVIEW_THRESHOLD
        )

        return ReviewResult(
            confidence=score,
            needs_human_review=needs_review,
            human_review_status=ReviewStatus.PENDING if needs_review else ReviewStatus.APPROVED
        )

    def request_human_review(self, query: str, answer: str) -> dict:
        return {
            "task_id": f"review_{hash(query + answer) % 100000}",
            "query": query,
            "answer": answer,
            "status": "pending",
            "timestamp": None,
            "reviewer": None,
            "comment": ""
        }