from typing import List
from langchain_core.documents import Document


class ReRanker:
    """Cross-Encoder重排序"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base"):
        self.model_name = model_name
        self.model = None

    def _load_model(self):
        """加载重排序模型"""
        try:
            from sentence_transformers import CrossEncoder

            print(f"加载重排序模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
        except Exception as e:
            print(f"加载失败，降级为无重排序: {e}")
            self.model = None

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """对检索结果重排序"""
        if not self.model or len(documents) <= top_n:
            return documents[:top_n]

        # 准备输入：(query, doc) 对
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 排序
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        # 添加元数据
        reranked_docs = []
        for idx, (doc, score) in enumerate(scored_docs[:top_n]):
            doc.metadata["rerank_score"] = float(score)
            doc.metadata["rerank_rank"] = idx + 1
            reranked_docs.append(doc)

        print(f"重排序完成：Top-{top_n}，最高分: {scored_docs[0][1]:.3f}")
        return reranked_docs