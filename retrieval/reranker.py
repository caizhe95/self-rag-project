# retrieval/reranker.py
from typing import List
from langchain_core.documents import Document


class ReRanker:
    """Cross-Encoder重排序（面试级）"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-base", enabled: bool = True):
        """
        初始化重排序器

        Args:
            model_name: 重排序模型名称
            enabled: 是否启用
        """
        self.model_name = model_name
        self.enabled = enabled
        self.model = None
        self._model_loaded = False

        if not enabled:
            print("⚠️  重排序功能已手动禁用")
            return

    def _load_model(self):
        """延迟加载模型（面试亮点：避免启动时慢）"""
        if self._model_loaded or not self.enabled:
            return

        try:
            from sentence_transformers import CrossEncoder

            print(f"📦 加载重排序模型: {self.model_name}")
            self.model = CrossEncoder(self.model_name, max_length=512)
            self._model_loaded = True
        except Exception as e:
            print(f"⚠️  重排序加载失败，降级为无重排序: {e}")
            self.enabled = False

    def rerank(self, query: str, documents: List[Document], top_n: int = 3) -> List[Document]:
        """重排序文档"""
        # 未启用或文档不足
        if not self.enabled or len(documents) <= top_n:
            return documents[:top_n]

        # 确保模型已加载
        if not self._model_loaded:
            self._load_model()

        if not self.model:
            return documents[:top_n]

        # 准备输入对
        pairs = [[query, doc.page_content] for doc in documents]

        # 计算分数
        scores = self.model.predict(pairs)

        # 排序并添加元数据
        scored_docs = list(zip(documents, scores))
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        reranked_docs = []
        for idx, (doc, score) in enumerate(scored_docs[:top_n]):
            doc.metadata.update({
                "rerank_score": float(score),
                "rerank_rank": idx + 1,
                "confidence": float(score)
            })
            reranked_docs.append(doc)

        print(f"✅ 重排序完成：Top-{top_n}，最高分: {scored_docs[0][1]:.3f}")
        return reranked_docs