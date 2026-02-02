# retrieval/hybrid_retriever.py
import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from retrieval.reranker import ReRanker


class HybridRetriever:
    """混合检索：修复分数传递问题"""

    def __init__(self, vector_store, documents: List[Document], config):
        self.vector_store = vector_store
        self.config = config
        self.documents = documents

        # BM25检索器（延迟初始化）
        self.bm25_retriever = self._init_bm25(documents) if documents else None

        # 重排序器
        self.reranker = ReRanker(
            model_name=getattr(config, "reranker_model", "BAAI/bge-reranker-base"),
            enabled=config.reranker_enabled
        )
        self._model_loaded = False

    def _init_bm25(self, docs: List[Document]) -> BM25Retriever:
        """初始化BM25（分离方法便于更新）"""
        retriever = BM25Retriever.from_documents(documents=docs)
        retriever.k = self.config.top_k
        print(f"✅ BM25检索器已初始化，文档数: {len(docs)}")
        return retriever

    async def aload_model(self):
        """延迟加载重排序模型"""
        if not self._model_loaded and self.config.reranker_enabled:
            await asyncio.to_thread(self.reranker._load_model)
            self._model_loaded = True

    async def aretrieve_with_cache(self, query: str) -> List[Document]:
        """异步检索"""
        await self.aload_model()

        # 降级处理：如果没有BM25，使用纯向量
        if self.bm25_retriever is None:
            print("⚠️  BM25未初始化，回退到纯向量检索")
            return await self.vector_store.asimilarity_search(query, k=self.config.top_k)

        return await self.ahybrid_retrieve(query)

    async def ahybrid_retrieve(self, query: str) -> List[Document]:
        """真正的混合检索逻辑 - 修复分数提取"""
        # 使用 similarity_search_with_score 获取分数
        vector_task = asyncio.to_thread(
            self.vector_store.similarity_search_with_score,  # 关键：带分数的API
            query,
            k=self.config.top_k * 2
        )

        bm25_task = asyncio.to_thread(
            self.bm25_retriever.invoke,
            query
        )

        # 等待结果（vector_docs_scores 是 (doc, score) 元组列表）
        vector_docs_scores, bm25_docs = await asyncio.gather(vector_task, bm25_task)

        # 解构文档和分数
        vector_docs = []
        for doc, score in vector_docs_scores:
            # 将分数添加到metadata
            doc.metadata["vector_score"] = float(score)
            doc.metadata["raw_score"] = float(score)  # 调试用
            doc.metadata["score_source"] = "vector"
            vector_docs.append(doc)

        # 合并与重排序
        merged_docs = self._merge_results(vector_docs, bm25_docs)

        if self.config.reranker_enabled:
            return await asyncio.to_thread(
                self.reranker.rerank, query, merged_docs, self.config.top_k
            )

        return merged_docs[:self.config.top_k]

    def _merge_results(self, vector_docs: List[Document], bm25_docs: List[Document]) -> List[Document]:
        """合并向量检索和BM25结果 - 修复分数存储"""
        weights = self.config.hybrid_weights

        # 处理向量文档的分数（已在上一步设置）
        for doc in vector_docs:
            if "vector_score" not in doc.metadata:
                # 降级处理：如果分数丢失，使用默认值
                doc.metadata["vector_score"] = 0.6 * weights["vector"]

        # 处理BM25文档的分数
        for doc in bm25_docs:
            raw_score = doc.metadata.get("score", 0.5)  # BM25的score在metadata中
            doc.metadata["bm25_score"] = float(raw_score) * weights["bm25"]

        # 去重合并
        merged = []
        seen = set()
        for doc in vector_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                merged.append(doc)

        return merged