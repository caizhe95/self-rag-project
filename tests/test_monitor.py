# core/test_monitor.py
import functools
import time
from typing import Callable, Any


def monitor_retrieval(func: Callable) -> Callable:
    """监控检索质量"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        # 获取query
        query = kwargs.get("query", "") or (args[1] if len(args) > 1 else "")

        start = time.perf_counter()
        docs = await func(*args, **kwargs)
        duration = time.perf_counter() - start

        # 计算相关性分数
        if docs and query:
            scores = []
            for doc in docs:
                score = doc.metadata.get("rerank_score") or \
                        doc.metadata.get("similarity", 0.0) or \
                        doc.metadata.get("hybrid_score", 0.0)
                if isinstance(score, (int, float)):
                    scores.append(float(score))

            relevance = max(scores) if scores else 0.0
        else:
            relevance = 0.0

        # 记录指标
        TestMetrics.record("retrieval", {
            "query": query,
            "duration": duration,
            "relevance": relevance,
            "docs_count": len(docs)
        })

        return docs

    return wrapper


def monitor_evaluation(func: Callable) -> Callable:
    """监控评估质量"""

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        review = func(*args, **kwargs)

        TestMetrics.record("evaluation", {
            "confidence": review.confidence,
            "hallucination_risk": review.hallucination_risk,
            "needs_human_review": review.needs_human_review
        })

        return review

    return wrapper


class TestMetrics:
    """全局测试指标存储"""
    data = {"retrieval": [], "evaluation": []}

    @classmethod
    def record(cls, category: str, metrics: dict):
        cls.data[category].append(metrics)

    @classmethod
    def get_report(cls) -> dict:
        if not cls.data["retrieval"] or not cls.data["evaluation"]:
            return {"total_queries": 0}

        return {
            "retrieval_avg_relevance": sum(r["relevance"] for r in cls.data["retrieval"]) / len(cls.data["retrieval"]),
            "retrieval_avg_duration": sum(r["duration"] for r in cls.data["retrieval"]) / len(cls.data["retrieval"]),
            "evaluation_avg_confidence": sum(e["confidence"] for e in cls.data["evaluation"]) / len(cls.data["evaluation"]),
            "total_queries": len(cls.data["evaluation"])
        }

    @classmethod
    def reset(cls):
        cls.data = {"retrieval": [], "evaluation": []}