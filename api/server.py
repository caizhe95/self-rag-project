# server.py
import os
import time
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from core.rag_chain import SelfRAGChain
from core.agent_graph import SelfRAGAgent
from config.setting import RAGConfig
from storage.knowledge_source import FileSystemSource, KnowledgeManager


# ==================== Pydantic 模型 ====================

class QueryRequest(BaseModel):
    """查询请求"""
    question: str
    session_id: Optional[str] = None
    chat_history: Optional[List[Dict[str, str]]] = None
    files: Optional[List[str]] = None


class ReviewActionRequest(BaseModel):
    """审核请求"""
    task_id: str
    action: str  # approved/rejected/modified
    modified_answer: Optional[str] = None
    comment: Optional[str] = None
    reviewer: Optional[str] = None


class ReviewResponse(BaseModel):
    """审核响应"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None


class RetrievalConfigRequest(BaseModel):
    """检索配置请求（AB测试用）"""
    hybrid_weights: Dict[str, float] = {"bm25": 0.4, "vector": 0.6}
    reranker_enabled: bool = True


# ==================== 生产监控数据存储 ====================

@dataclass
class QueryMetrics:
    """单次查询指标"""
    timestamp: float
    query: str
    model: str
    iteration_count: int
    confidence: float
    hallucination_risk: float
    retrieval_duration_ms: float
    total_duration_ms: float
    docs_count: int
    review_triggered: bool
    status: str  # success/error


class ProductionMonitor:
    """生产环境监控 - 内存存储（面试时可扩展为Redis/DB）"""
    MAX_HISTORY = 1000  # 保留最近1000条

    def __init__(self):
        self.history: List[QueryMetrics] = []
        self.total_queries = 0
        self.error_count = 0
        self.review_triggered_count = 0

    def record(self, metrics: QueryMetrics):
        """记录查询指标"""
        self.history.append(metrics)
        self.total_queries += 1

        if metrics.status == "error":
            self.error_count += 1
        if metrics.review_triggered:
            self.review_triggered_count += 1

        # 限制历史长度
        if len(self.history) > self.MAX_HISTORY:
            self.history.pop(0)

    def get_dashboard_data(self) -> Dict[str, Any]:
        """获取仪表盘数据"""
        if not self.history:
            return {"status": "no_data"}

        recent = self.history[-100:]  # 最近100条

        avg_confidence = sum(m.confidence for m in recent) / len(recent)
        avg_hallucination = sum(m.hallucination_risk for m in recent) / len(recent)
        avg_duration = sum(m.total_duration_ms for m in recent) / len(recent)

        # 模型分布
        model_stats = {}
        for m in recent:
            model_stats[m.model] = model_stats.get(m.model, 0) + 1

        return {
            "status": "healthy",
            "overview": {
                "total_queries": self.total_queries,
                "recent_queries": len(recent),
                "error_rate": self.error_count / max(self.total_queries, 1),
                "review_trigger_rate": self.review_triggered_count / max(self.total_queries, 1),
                "avg_confidence": round(avg_confidence, 2),
                "avg_hallucination_risk": round(avg_hallucination, 2),
                "avg_response_time_ms": round(avg_duration, 1),
            },
            "model_distribution": model_stats,
            "recent_history": [asdict(m) for m in recent[-10:]]  # 最近10条详情
        }

    def get_alerts(self) -> List[Dict[str, Any]]:
        """获取告警（置信度<0.5 或 幻觉>0.6）"""
        alerts = []
        for m in self.history[-50:]:  # 检查最近50条
            if m.confidence < 0.5:
                alerts.append({
                    "type": "low_confidence",
                    "timestamp": m.timestamp,
                    "query": m.query[:50],
                    "confidence": m.confidence,
                    "severity": "warning"
                })
            if m.hallucination_risk > 0.6:
                alerts.append({
                    "type": "high_hallucination",
                    "timestamp": m.timestamp,
                    "query": m.query[:50],
                    "risk": m.hallucination_risk,
                    "severity": "critical"
                })
        return alerts


# 创建全局监控实例
monitor = ProductionMonitor()


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 - 智能切换知识源"""
    global rag_chain, agent

    print("🚀 初始化 Self-RAG 系统...")

    vector_db_path = Path("data/chroma_db")
    if vector_db_path.exists():
        print(f"⚠️  检测到旧向量数据库，正在清理...")
        shutil.rmtree(vector_db_path)
        print(f"✅ 已清理 {vector_db_path}")

    config = RAGConfig()
    rag_chain = SelfRAGChain(config)

    # 检查运行模式
    is_test = os.getenv("LOCAL_TEST") == "true"

    if is_test:
        # ========== 测试模式：简单示例文档 ==========
        print("🧪 测试模式：加载示例文档")
        sample_docs = [
            {"text": "LangChain是一个用于开发LLM应用的框架...", "metadata": {"source": "langchain_intro.txt"}},
            {"text": "Self-RAG是RAG的增强版本...", "metadata": {"source": "self_rag_intro.txt"}}
        ]

        await rag_chain.aindex_documents(
            texts=[doc["text"] for doc in sample_docs],
            metadatas=[doc["metadata"] for doc in sample_docs]
        )
    else:
        # ========== 生产模式：加载 knowledge/ 目录 ==========
        print("🏭 生产模式：加载知识库文档...")

        knowledge_dir = Path("data/knowledge")
        if not knowledge_dir.exists():
            print(f"⚠️  知识库目录不存在: {knowledge_dir}")
            knowledge_dir.mkdir(parents=True, exist_ok=True)

        fs_source = FileSystemSource(path=str(knowledge_dir), priority=100)
        knowledge_manager = KnowledgeManager([fs_source])

        documents = await knowledge_manager.load_all_documents(deduplicate=True)
        print(f"📚 已加载 {len(documents)} 篇文档:")
        for doc in documents:
            print(f"   ✓ {doc.metadata.get('source', 'unknown')}")

        texts = [doc.page_content for doc in documents]
        metadatas = [doc.metadata for doc in documents]

        await rag_chain.aindex_documents(texts=texts, metadatas=metadatas)

    # 新增：初始化 Agent（复用已初始化的 rag_chain）
    print("🤖 初始化 Self-RAG Agent...")
    agent = SelfRAGAgent(rag_chain, config)
    print("✅ Agent 初始化完成")

    print(f"✅ 系统初始化完成 | 文档总数: {len(rag_chain.retriever.hybrid_retriever.documents)}")
    yield

    # 清理资源
    print("👋 系统关闭")


app = FastAPI(title="Self-RAG Agent API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7861", "http://localhost:7861"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain: Optional[SelfRAGChain] = None
agent: Optional[SelfRAGAgent] = None


# ==================== API 路由 ====================

@app.post("/api/query")
async def query(req: QueryRequest):
    """增强版查询接口 - 返回详细检索信息"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent 未初始化")

        # 记录开始时间
        start_time = time.time()
        retrieval_start = time.time()

        # 执行查询
        result = await agent.query(
            question=req.question,
            session_id=req.session_id or "default"
        )

        # 获取检索详情（复用已有结果）
        retrieval_info = await rag_chain.get_retrieval_info(req.question)
        retrieval_duration = (time.time() - retrieval_start) * 1000
        total_duration = (time.time() - start_time) * 1000

        # 构建详细sources
        sources = []
        for doc in retrieval_info.get("docs", []):
            sources.append({
                "source": doc.metadata.get("source", "unknown"),
                "vector_score": doc.metadata.get("vector_score"),
                "bm25_score": doc.metadata.get("bm25_score"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "hybrid_score": doc.metadata.get("hybrid_score"),
                "final_score": doc.metadata.get("final_score", 0),
                "content_preview": doc.page_content[:50] + "..."
            })

        return {
            "success": True,
            "data": {
                "answer": result["answer"],
                "confidence": result.get("confidence", 0),
                "iteration": result.get("iterations", 0),
                "sources": sources,
                "retrieval_metrics": retrieval_info.get("metrics", {}),
                "config_used": retrieval_info.get("config_used", {}),
                "timing": {
                    "retrieval_ms": retrieval_duration,
                    "total_ms": total_duration
                },
                "review_task_id": result.get("review_task_id"),
                "review_status": "pending" if result.get("needs_review") else None
            }
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# ==================== 监控端点 ====================

@app.get("/api/monitor/dashboard")
async def monitor_dashboard():
    """监控仪表盘数据"""
    return monitor.get_dashboard_data()


@app.get("/api/monitor/alerts")
async def monitor_alerts():
    """获取实时告警"""
    return {
        "alerts": monitor.get_alerts(),
        "alert_count": len(monitor.get_alerts())
    }


@app.get("/api/monitor/history")
async def monitor_history(limit: int = 100):
    """查询历史记录"""
    return {
        "history": [asdict(m) for m in monitor.history[-limit:]],
        "total": len(monitor.history)
    }


@app.get("/health")
async def health():
    """健康检查 - 增强版"""
    dashboard = monitor.get_dashboard_data()

    return {
        "status": "healthy",
        "mode": "agent",
        "model": rag_chain.config.llm_model if rag_chain else "unknown",
        "model_type": "大模型(32B)" if rag_chain and getattr(rag_chain.config, 'strict_mode', False) else "小模型(3B)",
        "document_count": len(
            rag_chain.retriever.hybrid_retriever.documents) if rag_chain and rag_chain.retriever else 0,
        "human_review_enabled": rag_chain.review_enabled if rag_chain else False,
        "pending_reviews": len(rag_chain.get_pending_reviews()) if rag_chain else 0,
        "monitor": dashboard.get("overview", {})
    }


# ==================== 审核相关接口（保持不变） ====================

@app.get("/api/reviews/pending")
async def get_pending_reviews():
    """获取待审核任务列表"""
    try:
        tasks = rag_chain.get_pending_reviews()
        return {
            "success": True,
            "count": len(tasks),
            "reviews": tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reviews/{task_id}")
async def get_review_detail(task_id: str):
    """获取审核任务详情"""
    try:
        detail = rag_chain.get_review_detail(task_id)
        if not detail:
            raise HTTPException(status_code=404, detail="审核任务不存在")

        return {
            "success": True,
            "review": detail
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reviews/submit", response_model=ReviewResponse)
async def submit_review(req: ReviewActionRequest, background_tasks: BackgroundTasks):
    """提交审核结果"""
    try:
        if req.action == "modified" and (not req.modified_answer or req.modified_answer.strip() == ""):
            return ReviewResponse(success=False, message="修改答案不能为空", data=None)

        success = await agent.submit_review(
            task_id=req.task_id,
            action=req.action,
            modified_answer=req.modified_answer
        )

        if not success:
            return ReviewResponse(success=False, message="审核提交失败", data=None)

        return ReviewResponse(
            success=True,
            message=f"审核已{req.action}，系统将继续执行",
            data={"task_id": req.task_id, "action": req.action}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reviews")
async def get_all_reviews(status: Optional[str] = None):
    """获取所有审核任务"""
    try:
        all_tasks = []
        for task_id, task in rag_chain.review_tasks.items():
            if status is None or task["status"] == status:
                all_tasks.append({
                    "task_id": task_id,
                    "status": task["status"],
                    "query": task["query"],
                    "confidence": task["confidence"],
                    "reviewer": task.get("reviewer"),
                    "reviewed_at": task.get("reviewed_at"),
                    "trigger_reason": task["trigger_reason"]
                })

        return {
            "success": True,
            "count": len(all_tasks),
            "reviews": all_tasks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==================== AB测试配置端点（可选） ====================

@app.post("/api/config/retrieval")
async def update_retrieval_config(req: RetrievalConfigRequest):
    """动态更新检索配置（AB测试用）"""
    try:
        if not rag_chain or not rag_chain.retriever:
            raise HTTPException(status_code=503, detail="检索器未初始化")

        await update_retrieval_config(
            hybrid_weights=req.hybrid_weights,
            reranker_enabled=req.reranker_enabled
        )

        return {
            "success": True,
            "message": "配置已更新",
            "config": {
                "hybrid_weights": req.hybrid_weights,
                "reranker_enabled": req.reranker_enabled
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/config/retrieval/reset")
async def reset_retrieval_config():
    """恢复原始检索配置"""
    try:
        if rag_chain:
            rag_chain.reset_retrieval_config()
        return {"success": True, "message": "配置已恢复"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/retrieval/debug")
async def debug_retrieval(query: str):
    """
    检索调试接口（返回详细信息，不生成答案）
    用于AB测试分析检索质量
    """
    try:
        if not rag_chain:
            raise HTTPException(status_code=503, detail="RAG链未初始化")

        result = await rag_chain.get_retrieval_info(query)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # 格式化返回
        docs_info = []
        for doc in result["docs"]:
            docs_info.append({
                "source": doc.metadata.get("source", "unknown"),
                "vector_score": doc.metadata.get("vector_score"),
                "bm25_score": doc.metadata.get("bm25_score"),
                "rerank_score": doc.metadata.get("rerank_score"),
                "hybrid_score": doc.metadata.get("hybrid_score"),
                "final_score": doc.metadata.get("final_score", 0),
                "content_preview": doc.page_content[:100] + "..."
            })

        return {
            "success": True,
            "query": query,
            "config_used": result["config_used"],
            "metrics": result["metrics"],
            "retrieved_docs": docs_info,
            "vector_count": len(result.get("vector_docs", [])),
            "bm25_count": len(result.get("bm25_docs", [])),
            "final_count": len(result["docs"])
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)