# core/retriever.py（修改版）

from typing import List, Dict, Any
from langchain_core.documents import Document
from retrieval.hybrid_retriever import HybridRetriever


class Retriever:
    """检索器：支持动态配置切换（AB测试用）"""

    def __init__(self, vector_store, documents: List[Document], config):
        self.vector_store = vector_store
        self.config = config
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            documents=documents,
            config=config
        )
        # 保存原始配置
        self.original_weights = getattr(config, "hybrid_weights", {"bm25": 0.4, "vector": 0.6})
        self.original_rerank = config.reranker_enabled

    async def retrieve(self, query: str) -> List[Document]:
        """标准检索接口"""
        return await self.hybrid_retriever.aretrieve_with_cache(query)

    async def retrieve_with_config(
            self,
            query: str,
            hybrid_weights: Dict[str, float] = None,
            use_reranker: bool = None
    ) -> Dict[str, Any]:
        """
        带配置的检索接口（AB测试用）

        Returns:
            包含docs和详细metrics的字典
        """
        return await self.hybrid_retriever.aretrieve_with_config(
            query,
            hybrid_weights=hybrid_weights,
            use_reranker=use_reranker
        )

    def update_config(
            self,
            hybrid_weights: Dict[str, float] = None,
            reranker_enabled: bool = None
    ):
        """动态更新配置"""
        if hybrid_weights is not None:
            self.hybrid_retriever.current_weights = hybrid_weights
            print(f"🔄 更新hybrid_weights: {hybrid_weights}")

        if reranker_enabled is not None:
            self.hybrid_retriever.reranker_enabled = reranker_enabled
            print(f"🔄 更新reranker_enabled: {reranker_enabled}")

    def reset_config(self):
        """恢复原始配置"""
        self.hybrid_retriever.current_weights = self.original_weights
        self.hybrid_retriever.reranker_enabled = self.original_rerank
        print("🔄 恢复原始配置")

    def update_documents(self, new_docs: List[Document]):
        """动态更新文档"""
        if not new_docs:
            return

        print(f"📚 正在更新BM25检索器，新增文档数: {len(new_docs)}")

        existing_docs = self.hybrid_retriever.documents
        seen_content = {doc.page_content for doc in existing_docs}
        unique_docs = [doc for doc in new_docs if doc.page_content not in seen_content]

        if unique_docs:
            existing_docs.extend(unique_docs)
            self.hybrid_retriever.bm25_retriever = self.hybrid_retriever._init_bm25(existing_docs)
            print(f"✅ BM25检索器已更新，总文档数: {len(existing_docs)}")