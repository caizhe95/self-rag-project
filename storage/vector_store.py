# storage/vector_store.py
from typing import List, Optional
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from config.setting import RAGConfig
from langchain_core.documents import Document


class VectorStoreManager:
    """向量存储管理器（支持文档暴露）"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = self._init_embeddings()
        self.vector_store: Optional[Chroma] = None

    def _init_embeddings(self) -> OllamaEmbeddings:
        """初始化 Ollama Embeddings"""
        return OllamaEmbeddings(
            model=self.config.embedding_model,
            base_url=self.config.ollama_base_url,
        )

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        """从文档创建向量数据库"""
        self.vector_store = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.vector_store_path,
            collection_name=self.config.collection_name
        )
        return self.vector_store

    def load_existing(self) -> Optional[Chroma]:
        """加载已有向量库"""
        try:
            self.vector_store = Chroma(
                persist_directory=self.config.vector_store_path,
                embedding_function=self.embeddings,
                collection_name=self.config.collection_name
            )
            return self.vector_store
        except Exception as e:
            print(f"❌ 加载向量库失败: {e}")
            return None

    def get_all_documents(self) -> List[Document]:
        """获取所有已存储的文档（为BM25准备）"""
        if not self.vector_store:
            return []

        try:
            # Chroma 获取所有文档
            results = self.vector_store.get(include=["documents", "metadatas"])
            docs = []
            for i, (content, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
                docs.append(Document(page_content=content, metadata=metadata or {"source": f"doc_{i}"}))
            return docs
        except Exception as e:
            print(f"⚠️  获取文档失败: {e}")
            return []

    def add_documents(self, documents: List[Document]) -> None:
        """动态添加文档（用于OCR）"""
        if not self.vector_store:
            raise ValueError("向量存储未初始化")

        self.vector_store.add_documents(documents)
        print(f"✅ 向量库已添加 {len(documents)} 个文档")