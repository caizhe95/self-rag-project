# retrieval/hybrid_retriever.py（完整修改版）

import asyncio
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

from retrieval.reranker import ReRanker


class HybridRetriever:
    """混合检索：支持详细指标输出"""

    def __init__(self, vector_store, documents: List[Document], config):
        self.vector_store = vector_store
        self.config = config
        self.documents = documents

        # BM25检索器
        self.bm25_retriever = self._init_bm25(documents) if documents else None

        # 重排序器
        self.reranker = ReRanker(
            model_name=getattr(config, "reranker_model", "BAAI/bge-reranker-base"),
            enabled=config.reranker_enabled
        )
        self._model_loaded = False

        # 运行时配置（支持动态切换）
        self.current_weights = getattr(config, "hybrid_weights", {"bm25": 0.4, "vector": 0.6})
        self.reranker_enabled = config.reranker_enabled

    def _init_bm25(self, docs: List[Document]) -> BM25Retriever:
        """初始化BM25"""
        retriever = BM25Retriever.from_documents(documents=docs)
        retriever.k = self.config.top_k
        print(f"✅ BM25检索器已初始化，文档数: {len(docs)}")
        return retriever

    async def aload_model(self):
        """延迟加载重排序模型"""
        if not self._model_loaded and self.reranker_enabled:
            await asyncio.to_thread(self.reranker._load_model)
            self._model_loaded = True

    async def aretrieve_with_config(
            self,
            query: str,
            hybrid_weights: Dict[str, float] = None,
            use_reranker: bool = None
    ) -> Dict[str, Any]:
        """
        带配置的检索接口（支持AB测试动态切换）

        Returns:
            {
                "docs": List[Document],  # 最终文档
                "vector_docs": List[Document],  # 向量检索原始结果
                "bm25_docs": List[Document],  # BM25原始结果
                "merged_docs": List[Document],  # 合并后去重
                "metrics": {
                    "vector_count": int,
                    "bm25_count": int,
                    "merged_count": int,
                    "final_count": int,
                    "vector_time_ms": float,
                    "bm25_time_ms": float,
                    "merge_time_ms": float,
                    "rerank_time_ms": float,
                }
            }
        """
        weights = hybrid_weights or self.current_weights
        use_rerank = use_reranker if use_reranker is not None else self.reranker_enabled

        await self.aload_model()

        metrics = {}
        start_total = asyncio.get_event_loop().time()

        # 降级处理
        if self.bm25_retriever is None or weights.get("bm25", 0) == 0:
            print("⚠️  BM25未启用或权重为0，使用纯向量检索")
            return await self._pure_vector_retrieve(query, metrics)

        # 1. 向量检索（带分数）
        vector_start = asyncio.get_event_loop().time()
        vector_docs_scores = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            k=self.config.top_k * 2
        )
        metrics["vector_time_ms"] = (asyncio.get_event_loop().time() - vector_start) * 1000

        # 解析向量和分数
        vector_docs = []
        for doc, score in vector_docs_scores:
            doc.metadata["vector_score"] = float(score)
            doc.metadata["score_source"] = "vector"
            vector_docs.append(doc)
        metrics["vector_count"] = len(vector_docs)

        # 2. BM25检索
        bm25_start = asyncio.get_event_loop().time()
        bm25_docs = await asyncio.to_thread(
            self.bm25_retriever.invoke,
            query
        )
        metrics["bm25_time_ms"] = (asyncio.get_event_loop().time() - bm25_start) * 1000

        # 提取BM25分数
        for doc in bm25_docs:
            raw_score = doc.metadata.get("score", 0.5)
            doc.metadata["bm25_score"] = float(raw_score)
            doc.metadata["score_source"] = "bm25"
        metrics["bm25_count"] = len(bm25_docs)

        # 3. 合并去重
        merge_start = asyncio.get_event_loop().time()
        merged_docs = self._merge_results(vector_docs, bm25_docs, weights)
        metrics["merge_time_ms"] = (asyncio.get_event_loop().time() - merge_start) * 1000
        metrics["merged_count"] = len(merged_docs)

        # 4. 重排序
        final_docs = merged_docs
        rerank_time = 0

        if use_rerank and self.reranker.enabled and len(merged_docs) > 1:
            rerank_start = asyncio.get_event_loop().time()
            final_docs = await asyncio.to_thread(
                self.reranker.rerank, query, merged_docs, self.config.top_k
            )
            rerank_time = (asyncio.get_event_loop().time() - rerank_start) * 1000
            # 标记重排序分数
            for doc in final_docs:
                doc.metadata["final_score"] = doc.metadata.get("rerank_score", 0)
        else:
            # 无重排序，取TopK
            final_docs = merged_docs[:self.config.top_k]
            for doc in final_docs:
                doc.metadata["final_score"] = max(
                    doc.metadata.get("vector_score", 0),
                    doc.metadata.get("bm25_score", 0)
                )

        metrics["rerank_time_ms"] = rerank_time
        metrics["final_count"] = len(final_docs)
        metrics["total_time_ms"] = (asyncio.get_event_loop().time() - start_total) * 1000

        return {
            "docs": final_docs,
            "vector_docs": vector_docs,
            "bm25_docs": bm25_docs,
            "merged_docs": merged_docs,
            "metrics": metrics
        }

    async def _pure_vector_retrieve(self, query: str, metrics: Dict) -> Dict[str, Any]:
        """纯向量检索"""
        start = asyncio.get_event_loop().time()

        docs = await asyncio.to_thread(
            self.vector_store.similarity_search_with_score,
            query,
            k=self.config.top_k
        )

        vector_docs = []
        for doc, score in docs:
            doc.metadata["vector_score"] = float(score)
            doc.metadata["final_score"] = float(score)
            vector_docs.append(doc)

        metrics["vector_time_ms"] = (asyncio.get_event_loop().time() - start) * 1000
        metrics["vector_count"] = len(vector_docs)
        metrics["bm25_time_ms"] = 0
        metrics["bm25_count"] = 0
        metrics["merge_time_ms"] = 0
        metrics["merged_count"] = len(vector_docs)
        metrics["rerank_time_ms"] = 0
        metrics["final_count"] = len(vector_docs)
        metrics["total_time_ms"] = metrics["vector_time_ms"]

        return {
            "docs": vector_docs,
            "vector_docs": vector_docs,
            "bm25_docs": [],
            "merged_docs": vector_docs,
            "metrics": metrics
        }

    def _merge_results(
            self,
            vector_docs: List[Document],
            bm25_docs: List[Document],
            weights: Dict[str, float]
    ) -> List[Document]:
        """合并向量检索和BM25结果 - 带权重归一化"""

        # 归一化分数（避免不同量纲）
        def normalize_scores(docs: List[Document], key: str, weight: float):
            if not docs:
                return
            scores = [d.metadata.get(key, 0) for d in docs]
            max_score = max(scores) if scores else 1.0
            min_score = min(scores) if scores else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0

            for doc in docs:
                raw = doc.metadata.get(key, 0)
                # Min-Max归一化到[0,1]后乘以权重
                normalized = (raw - min_score) / score_range if score_range > 0 else 0
                doc.metadata["hybrid_score"] = normalized * weight

        # 应用权重
        vector_weight = weights.get("vector", 0.6)
        bm25_weight = weights.get("bm25", 0.4)

        normalize_scores(vector_docs, "vector_score", vector_weight)
        normalize_scores(bm25_docs, "bm25_score", bm25_weight)

        # 合并去重（保留最高分）
        merged_map = {}
        for doc in vector_docs + bm25_docs:
            content = doc.page_content
            existing_doc = merged_map.get(content)
            existing_score = existing_doc.metadata.get("hybrid_score", 0) if existing_doc else 0
            if content not in merged_map or doc.metadata.get("hybrid_score", 0) > existing_score:
                merged_map[content] = doc

        # 按混合分数排序
        merged = list(merged_map.values())
        merged.sort(key=lambda x: x.metadata.get("hybrid_score", 0), reverse=True)

        return merged

    async def aretrieve_with_cache(self, query: str) -> List[Document]:
        """兼容原有接口"""
        result = await self.aretrieve_with_config(query)
        return result["docs"]