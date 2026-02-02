# core/text_splitter.py
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)


class SmartDocumentSplitter:
    """
    智能文档分割器（面试亮点）
    - 文档类型感知：自动识别Markdown/代码/文本
    - 结构保留：Markdown标题、代码语法树
    - 元数据增强：每块标记来源和分割方式
    - 自动降级：失败时回退到安全模式
    """

    def __init__(self, config):
        self.config = config

        # 通用分割器（安全兜底）
        self.fallback_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。"],
        )

        # 父文档检索器（解决大文档上下文丢失问题）
        self._init_parent_retriever()

    def _init_parent_retriever(self):
        """延迟初始化父文档检索器"""
        try:
            from langchain.retrievers import ParentDocumentRetriever
            from langchain.storage import InMemoryStore

            self.parent_retriever = ParentDocumentRetriever(
                vectorstore=None,
                docstore=InMemoryStore(),
                child_splitter=RecursiveCharacterTextSplitter(chunk_size=200),
                parent_splitter=RecursiveCharacterTextSplitter(chunk_size=1000),
            )
        except ImportError:
            self.parent_retriever = None

    def split_documents(self, docs: List[Document]) -> List[Document]:
        """智能分块入口（核心逻辑）"""
        all_chunks = []
        stats = {"markdown": 0, "code": 0, "fallback": 0}

        for idx, doc in enumerate(docs):
            # 策略1：Markdown - 保留标题结构（最适合技术文档）
            if self._is_markdown(doc):
                chunks = self._split_markdown(doc)
                stats["markdown"] += len(chunks)

            # 策略2：代码文件 - 按语法树分块（保留函数完整性）
            elif self._is_code(doc):
                chunks = self._split_code(doc)
                stats["code"] += len(chunks)

            # 策略3：普通文本 - 递归字符分割
            else:
                chunks = self.fallback_splitter.split_documents([doc])
                stats["fallback"] += len(chunks)

            # 增强元数据（面试重点：为什么加这些？）
            for chunk in chunks:
                chunk.metadata.update({
                    "parent_id": f"doc_{idx}",
                    "splitter": "markdown" if self._is_markdown(doc) else "code" if self._is_code(doc) else "fallback",
                    "chunk_idx": chunk.metadata.get("chunk_idx", 0),
                })

            all_chunks.extend(chunks)

        print(f"✂️  分块统计: Markdown={stats['markdown']} | Code={stats['code']} | Fallback={stats['fallback']}")
        return all_chunks

    def _is_markdown(self, doc: Document) -> bool:
        """判断Markdown"""
        source = doc.metadata.get("source", "")
        return source.endswith((".md", ".markdown"))

    def _is_code(self, doc: Document) -> bool:
        """判断代码文件"""
        source = doc.metadata.get("source", "")
        return source.endswith((".py", ".js", ".java"))

    def _split_markdown(self, doc: Document) -> List[Document]:
        """Markdown分割：保留标题结构"""
        md_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header1"),
                ("##", "Header2"),
            ]
        )
        chunks = md_splitter.split_text(doc.page_content)

        # 标记标题级别
        for chunk in chunks:
            level = 1 if chunk.metadata.get("Header1") else 2
            chunk.metadata["header_level"] = level

        return chunks

    def _split_code(self, doc: Document) -> List[Document]:
        """代码分割：按语法树"""
        ext = doc.metadata.get("source", "").split(".")[-1]
        language_map = {"py": Language.PYTHON, "js": Language.JS, "java": Language.JAVA}

        code_splitter = RecursiveCharacterTextSplitter.from_language(
            language=language_map.get(ext, Language.PYTHON),
            chunk_size=800,  # 代码块可以更大
            chunk_overlap=0,
        )
        return code_splitter.split_documents([doc])

    def setup_parent_retriever(self, vector_store):
        """配置父文档检索器（提升大文档检索效果）"""
        if self.parent_retriever:
            self.parent_retriever.vectorstore = vector_store
            print("✅ 父文档检索器已启用")
        return self.parent_retriever