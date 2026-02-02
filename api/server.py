# server.py
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Dict, Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import shutil
from core.rag_chain import SelfRAGChain
from core.agent_graph import SelfRAGAgent  # 新增：导入 Agent
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


# ==================== 生命周期管理 ====================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期 - 智能切换知识源"""
    global rag_chain, agent  # 新增：声明 agent 为全局

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


app = FastAPI(title="Self-RAG Agent API", lifespan=lifespan)  # 修改：标题改为 Agent

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:7861", "http://localhost:7861"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_chain: Optional[SelfRAGChain] = None
agent: Optional[SelfRAGAgent] = None  # 新增：全局 agent 变量


# ==================== API 路由 ====================

@app.post("/api/query")
async def query(req: QueryRequest):
    """统一查询接口 - 使用 Agent"""
    try:
        if not agent:
            raise HTTPException(status_code=503, detail="Agent 未初始化")

        # 使用 Agent 进行查询（自动处理工具调用和 Self-RAG 流程）
        result = await agent.query(
            question=req.question,
            session_id=req.session_id or "default"
        )

        # 转换返回格式以兼容现有前端（webui_user.py 等）
        return {
            "success": True,
            "data": {
                "answer": result["answer"],
                "confidence": result["confidence"],
                "iteration": result["iterations"],  # Agent 返回的是 iterations（复数）
                "sources": [],  # Agent 模式可能需要从状态中提取，这里留空或后续扩展
                "review_task_id": result.get("review_task_id"),
                "review_status": "pending" if result.get("needs_review") else None
            }
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    """健康检查 - 显示当前模型配置"""
    config_info = {
        "status": "healthy",
        "mode": "agent",  # 修改：标识为 agent 模式
        "model": rag_chain.config.llm_model if rag_chain else "unknown",
        "model_type": "大模型(32B)" if rag_chain and getattr(rag_chain.config, 'strict_mode', False) else "小模型(3B)",
        "document_count": len(
            rag_chain.retriever.hybrid_retriever.documents) if rag_chain and rag_chain.retriever else 0,
        "human_review_enabled": rag_chain.review_enabled if rag_chain else False,
        "pending_reviews": len(rag_chain.get_pending_reviews()) if rag_chain else 0
    }
    return config_info


# ==================== 审核相关接口（保持不变，直接操作 rag_chain） ====================

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

        # 通过 Agent 提交审核（Agent 内部会 resume 图执行）
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


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)