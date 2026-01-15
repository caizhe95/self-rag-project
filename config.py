# config.py
import os
from dataclasses import dataclass


@dataclass
class Config:
    # Ollama服务
    OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    # 模型配置（已替换为bge-m3）
    EMBEDDING_MODEL: str = "bge-m3:latest"
    LLM_MODEL: str = "deepseek-coder:33b"

    # 向量数据库
    VECTOR_STORE_PATH: str = "./data/chroma_db"
    COLLECTION_NAME: str = "knowledge_base"

    # 检索配置
    TOP_K: int = 5
    HYBRID_WEIGHTS: dict = None

    # Self-RAG配置
    CONFIDENCE_THRESHOLD: float = 0.6
    MAX_ITERATIONS: int = 2

    # 重排序配置
    RERANKER_ENABLED: bool = True
    RERANKER_MODEL: str = "BAAI/bge-reranker-base"
    RERANKER_TOP_N: int = 3

    # 人工审核
    HUMAN_REVIEW_ENABLED: bool = True
    HUMAN_REVIEW_THRESHOLD: float = 0.5  # 从0.4调高，让演示更稳定

    # 流式输出
    STREAMING_ENABLED: bool = True

    # 缓存配置
    CACHE_ENABLED: bool = True
    CACHE_MAX_SIZE: int = 100
    CACHE_TTL: int = 1800  # 30分钟过期

    def __post_init__(self):
        if self.HYBRID_WEIGHTS is None:
            self.HYBRID_WEIGHTS = {"bm25": 0.4, "vector": 0.6}


config = Config()