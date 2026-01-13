import asyncio
from typing import List, Dict, Optional, AsyncIterator, Any
from config import Config
from database.vector_store import VectorStoreManager
from retrieval.hybrid_retriever import HybridRetriever
from evaluation.self_evaluator import SelfEvaluator
from workflow.rag_graph import RAGWorkflow, RAGState
from langchain_community.llms import OllamaLLM
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


class SelfRAGChain:
    """Self-RAG主链"""

    def __init__(self, config: Config):
        self.config = config
        self.llm = self._init_llm()

        self.vector_manager = VectorStoreManager(config)
        self.retriever = None
        self.evaluator = SelfEvaluator(self.llm, config)

    def _init_llm(self):
        return OllamaLLM(
            model=self.config.LLM_MODEL,
            base_url=self.config.OLLAMA_BASE_URL,
            temperature=0.1,
        )

    async def aindex_documents(self, texts: List[str], metadatas: Optional[List[Dict]] = None):
        """索引文档"""
        docs = [
            Document(page_content=text, metadata=meta or {"source": f"doc_{i}"})
            for i, (text, meta) in enumerate(zip(texts, metadatas or [None] * len(texts)))
        ]

        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        chunks = await asyncio.to_thread(splitter.split_documents, docs)

        # 创建向量存储
        vector_store = await asyncio.to_thread(
            self.vector_manager.create_from_documents, chunks
        )

        # 加载重排序模型
        self.retriever = HybridRetriever(
            vector_store=vector_store,
            documents=chunks,
            config=self.config
        )

        if hasattr(self.retriever, 'reranker'):
            await asyncio.to_thread(self.retriever.reranker._load_model)

        self.workflow = RAGWorkflow(
            retriever=self.retriever,
            evaluator=self.evaluator,
            llm=self.llm,
            config=self.config
        )

    async def astream_query(self, question: str) -> AsyncIterator[Dict[str, Any]]:
        """流式查询"""
        if not hasattr(self, 'workflow'):
            raise ValueError("请先调用aindex_documents()索引文档")

        print(f"\n{'=' * 60}")
        print(f"Self-RAG异步查询: {question}")
        print('=' * 60)

        initial_state: RAGState = {
            "query": question,
            "documents": [],
            "answer": "",
            "sources": [],
            "confidence": 0.0,
            "iteration": 0,
            "review_result": None,
            "human_review_decision": None,
            "human_review_comment": ""
        }

        # 流式事件
        async for event in self.workflow.graph.astream_events(initial_state, version="v1"):
            yield event

    def query(self, question: str) -> Dict:
        """兼容方法"""
        import asyncio
        return asyncio.run(self._run_sync(question))

    async def _run_sync(self, question: str) -> Dict:
        """辅助同步执行"""
        async for event in self.astream_query(question):
            if event["event"] == "on_chain_end" and event["name"] == "finalize":
                return event["data"]["output"]
        return {}