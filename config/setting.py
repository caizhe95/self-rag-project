# config/setting.py - 适配 llama3.2:3b(本地) 与 deepseek-r1:32b(云端3090)
import os
from dataclasses import dataclass, field
from typing import Literal, Dict, Any
from dotenv import load_dotenv

load_dotenv()


@dataclass
class RAGConfig:
    """Self-RAG 配置：自动识别 llama3.2:3b(小模型) 与 deepseek-r1:32b(大模型)"""

    # ==================== 模型配置（通过 .env 切换）====================
    # 本地调试：LLM_MODEL=llama3.2:3b
    # 云端3090：LLM_MODEL=deepseek-r1:32b
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "all-minilm:latest")
    temperature: float = 0.1

    # ==================== 策略自适应字段 ====================
    chunk_size: int = field(init=False)
    chunk_overlap: int = field(init=False)
    top_k: int = field(init=False)
    max_iterations: int = field(init=False)
    use_llm_contradiction: bool = field(init=False)
    extract_claims_max: int = field(init=False)
    strict_mode: bool = field(init=False)

    # ==================== 其他配置 ====================
    ocr_enabled: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    ocr_language: str = "chi_sim+eng"
    search_type: Literal["similarity", "mmr"] = "similarity"
    reranker_enabled: bool = os.getenv("RERANKER_ENABLED", "true").lower() == "true"
    reranker_model: str = "BAAI/bge-reranker-base"
    hybrid_weights: Dict[str, float] = field(default_factory=lambda: {"bm25": 0.4, "vector": 0.6})
    confidence_threshold: float = 0.6
    retrieval_relevance_threshold: float = 0.2
    human_review_enabled: bool = os.getenv("HUMAN_REVIEW_ENABLED", "true").lower() == "true"
    human_review_threshold: float = float(os.getenv("HUMAN_REVIEW_THRESHOLD", "0.4"))
    vector_store_path: str = "./data/chroma_db"
    collection_name: str = "knowledge_base"

    def __post_init__(self):
        """自动识别固定模型配置"""
        model = self.llm_model.lower()

        # ==================== 大模型：deepseek-r1:32b (云端3090) ====================
        # 适配24GB显存：32B-Q4量化约20-22GB，留2-4GB余量给KV Cache
        if "deepseek" in model and ("32b" in model or "33b" in model):
            self.chunk_size = 600  # 保守设置，避免3090爆显存（原文档建议800，但600更安全）
            self.chunk_overlap = 60  # 10%重叠
            self.top_k = 6  # 3090可承受6篇文档召回（平衡精度与显存）
            self.max_iterations = 3  # 支持Self-RAG深度优化

            # 评估器：启用LLM矛盾检测（32B能力足够）
            self.use_llm_contradiction = True
            self.extract_claims_max = 5
            self.strict_mode = True

            print(f"⚙️ 云端3090配置: {self.llm_model}")
            print(f"   显存优化: chunk={self.chunk_size}, top_k={self.top_k} (适配24GB显存)")

        # ==================== 小模型：llama3.2:3b (本地笔记本) ====================
        else:
            self.chunk_size = 300  # 2K上下文，小分块防溢出
            self.chunk_overlap = 30  # 10%重叠
            self.top_k = 3  # 3篇精准召回，防信息过载
            self.max_iterations = 2  # 2次迭代，保证笔记本响应速度

            # 评估器：规则检测（快速不超时）
            self.use_llm_contradiction = False
            self.extract_claims_max = 2
            self.strict_mode = False

            print(f"⚙️ 本地开发配置: {self.llm_model}")
            print(f"   轻量模式: chunk={self.chunk_size}, top_k={self.top_k} (适配笔记本CPU/GPU)")