import asyncio
from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from retrieval.reranker import ReRanker


class HybridRetriever:
    """混合检索（BM25 + 向量 + Rerank）"""

    def __init__(self, vector_store, documents: List[Document], config):
        self.vector_store = vector_store
        self.config = config
        self.bm25_retriever = BM25Retriever.from_documents(documents=documents)
        self.bm25_retriever.k = config.TOP_K

        # 重排序器
        self.reranker = ReRanker(
            model_name=getattr(config, "RERANKER_MODEL", "BAAI/bge-reranker-base")
        )

        # 加载重排序模型
        self._model_loaded = False

    async def aload_model(self):
        """加载模型"""
        if not self._model_loaded and self.config.RERANKER_ENABLED:
            if hasattr(self, 'reranker'):
                await asyncio.to_thread(self.reranker._load_model)
                self._model_loaded = True

    def retrieve_with_cache(self, query: str) -> List[Document]:
        """同步版本（向后兼容）"""
        import asyncio
        return asyncio.run(self.aretrieve_with_cache(query))

    async def aretrieve_with_cache(self, query: str) -> List[Document]:
        """异步缓存版本"""
        # 先确保模型加载
        if self.config.RERANKER_ENABLED:
            await self.aload_model()

        return await self.ahybrid_retrieve(query)

    async def ahybrid_retrieve(self, query: str) -> List[Document]:
        """混合检索"""
        # 并行执行向量检索和BM25
        vector_task = asyncio.to_thread(
            self.vector_store.similarity_search, query, k=self.config.TOP_K * 2
        )
        bm25_task = asyncio.to_thread(
            self.bm25_retriever.get_relevant_documents, query
        )

        vector_docs, bm25_docs = await asyncio.gather(vector_task, bm25_task)

        # 去重合并
        all_docs = list({doc.page_content: doc for doc in (vector_docs + bm25_docs)}.values())

        # 重排序
        if self.config.RERANKER_ENABLED:
            reranked_docs = await asyncio.to_thread(
                self.reranker.rerank, query, all_docs, self.config.TOP_K
            )
        else:
            reranked_docs = all_docs[:self.config.TOP_K]

        for doc in reranked_docs:
            doc.metadata[
                "retrieval_method"] = "async_hybrid_reranked" if self.config.RERANKER_ENABLED else "async_hybrid"

        return reranked_docs