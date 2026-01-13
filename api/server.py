# api/server.py - 完整修复版
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import json
import uuid
from core.rag_chain import SelfRAGChain
from config import Config

app = FastAPI(title="Self-RAG API")

# 添加生产环境CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str


rag_chain = None


@app.on_event("startup")
async def startup():
    global rag_chain
    config = Config()
    rag_chain = SelfRAGChain(config)

    # 确保数据目录存在
    import os
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)

    # 异步索引示例文档
    SAMPLE_DOCS = [
        {"text": "LangChain是一个框架...", "metadata": {"source": "doc1"}},
        {"text": "Self-RAG是增强版RAG...", "metadata": {"source": "doc2"}}
    ]
    await rag_chain.aindex_documents(
        texts=[doc["text"] for doc in SAMPLE_DOCS],
        metadatas=[doc["metadata"] for doc in SAMPLE_DOCS]
    )
    print("系统初始化完成")


@app.post("/api/query")
async def query(req: QueryRequest, stream: bool = Query(False)):
    """统一查询接口"""
    if stream:
        async def generate():
            async for event in rag_chain.astream_query(req.question):
                if event["event"] == "on_chain_stream" and event["name"] == "generate":
                    yield f"data: {json.dumps({'text': event['data']['chunk']})}\n\n"
            yield f"data: {json.dumps({'status': 'done'})}\n\n"

        return StreamingResponse(generate(), media_type="text/plain")

    # 同步查询
    result = await rag_chain._run_sync(req.question)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "iteration": result["iteration"],
        "sources": result["sources"],
        "needs_review": result.get("review_result", {}).get("needs_human_review", False),
        "review_task_id": str(uuid.uuid4()) if result.get("review_result", {}).get("needs_human_review") else None
    }


@app.post("/api/stream/{question}")
async def stream_query(question: str):
    async def generate():
        async for event in rag_chain.astream_query(question):
            if event["event"] == "on_chain_stream" and event["name"] == "generate":
                yield f"data: {json.dumps({'text': event['data']['chunk']})}\n\n"
        yield f"data: {json.dumps({'status': 'done'})}\n\n"

    return StreamingResponse(generate(), media_type="text/plain")


@app.get("/health")
async def health():
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)