from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma


class VectorStoreManager:
    def __init__(self, config):
        self.config = config
        self.embeddings = self._init_embeddings()

    def _init_embeddings(self):
        return OllamaEmbeddings(
            model=self.config.EMBEDDING_MODEL,
            base_url=self.config.OLLAMA_BASE_URL,
        )

    def create_from_documents(self, documents: List[Document]) -> Chroma:
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.config.VECTOR_STORE_PATH,
            collection_name=self.config.COLLECTION_NAME
        )

    def load_existing(self) -> Optional[Chroma]:
        try:
            return Chroma(
                persist_directory=self.config.VECTOR_STORE_PATH,
                embedding_function=self.embeddings,
                collection_name=self.config.COLLECTION_NAME
            )
        except Exception as e:
            print(f"加载向量库失败: {e}")
            return None