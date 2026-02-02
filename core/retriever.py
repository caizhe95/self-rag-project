# core/retriever.py
from typing import List
from langchain_core.documents import Document
from retrieval.hybrid_retriever import HybridRetriever


class Retriever:
    """检索器：支持动态文档更新"""

    def __init__(self, vector_store, documents: List[Document], config):
        self.vector_store = vector_store
        self.config = config
        self.hybrid_retriever = HybridRetriever(
            vector_store=vector_store,
            documents=documents,
            config=config
        )

    async def retrieve(self, query: str) -> List[Document]:
        """异步检索"""
        return await self.hybrid_retriever.aretrieve_with_cache(query)

    def update_documents(self, new_docs: List[Document]):
        """动态更新文档（索引后调用）"""
        if not new_docs:
            return

        print(f"📚 正在更新BM25检索器，新增文档数: {len(new_docs)}")

        # 获取现有文档
        existing_docs = self.hybrid_retriever.documents

        # 避免重复
        seen_content = {doc.page_content for doc in existing_docs}
        unique_docs = [doc for doc in new_docs if doc.page_content not in seen_content]

        if unique_docs:
            existing_docs.extend(unique_docs)
            # 重新初始化BM25
            self.hybrid_retriever.bm25_retriever = self.hybrid_retriever._init_bm25(existing_docs)
            print(f"✅ BM25检索器已更新，总文档数: {len(existing_docs)}")