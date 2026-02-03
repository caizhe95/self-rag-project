# test_monitor.py
import functools
import time
from typing import Callable, Any


def monitor_retrieval(func: Callable) -> Callable:
    """监控检索质量 - 装饰器模式"""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        query = kwargs.get("query", "") or (args[1] if len(args) > 1 else "")

        start = time.perf_counter()
        docs = await func(*args, **kwargs)
        duration = time.perf_counter() - start

        # 计算相关性
        if docs and query:
            scores = []
            for doc in docs:
                score = (doc.metadata.get("rerank_score") or
                        doc.metadata.get("similarity", 0.0) or
                        doc.metadata.get("hybrid_score", 0.0))
                if isinstance(score, (int, float)):
                    scores.append(float(score))
            relevance = max(scores) if scores else 0.0
        else:
            relevance = 0.0

        TestMetrics.record("retrieval", {
            "query": query,
            "duration": duration,
            "relevance": relevance,
            "docs_count": len(docs)
        })

        return docs

    return wrapper


def monitor_evaluation(func: Callable) -> Callable:
    """监控评估质量 - 装饰器模式"""

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
            "evaluation_avg_hallucination": sum(e["hallucination_risk"] for e in cls.data["evaluation"]) / len(cls.data["evaluation"]),
            "total_queries": len(cls.data["evaluation"])
        }

    @classmethod
    def get_detailed_report(cls) -> str:
        """生成格式化报告"""
        report = cls.get_report()
        if report["total_queries"] == 0:
            return "暂无监控数据"

        lines = [
            "\n" + "=" * 60,
            "📊 Self-RAG 性能监控报告",
            "=" * 60,
            f"📝 总查询次数: {report['total_queries']}",
            "",
            "【检索性能】",
            f"  ⏱️  平均耗时: {report['retrieval_avg_duration']*1000:.1f}ms",
            f"  🎯 平均相关性: {report['retrieval_avg_relevance']:.2f}",
            "",
            "【评估质量】",
            f"  ✅ 平均置信度: {report['evaluation_avg_confidence']:.2f}",
            f"  ⚠️  平均幻觉风险: {report['evaluation_avg_hallucination']:.2f}",
            "=" * 60
        ]
        return "\n".join(lines)

    @classmethod
    def reset(cls):
        cls.data = {"retrieval": [], "evaluation": []}