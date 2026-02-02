# core/document_processor.py（生产级实现）
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Any

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.ocr_processor import OCRProcessor
from storage.vector_store import VectorStoreManager


class DocumentProcessor:
    """文档处理器：加载、分块、OCR、向量化（生产级实现）"""

    def __init__(self, config):
        self.config = config

        # 智能分块器
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", ".", "!", "?", " ", ""]
        )

        # OCR处理器
        self.ocr_processor = OCRProcessor(config.ocr_language, config.ocr_enabled)

        # 向量存储管理器（延迟初始化）
        self.vector_manager: Optional[VectorStoreManager] = None
        self.vector_store: Optional[Any] = None

    def _load_text_documents(self, texts: List[str], metadatas: List[Dict]) -> List[Document]:
        """加载纯文本文档"""
        documents = []
        for i, text in enumerate(texts):
            metadata = metadatas[i] if i < len(metadatas) else {"source": f"doc_{i}"}
            documents.append(Document(page_content=text, metadata=metadata))
        return documents

    async def _load_file_documents(self, files: List[Path]) -> List[Document]:
        """加载文件文档（带OCR）"""
        if not files or not self.ocr_processor.is_available():
            return []

        print(f"📷 OCR处理：{len(files)}个文件")
        docs = []
        for file_path in files:
            text = await self.ocr_processor.extract_text(file_path)
            if text:
                doc = self.ocr_processor.create_document(file_path, text)
                docs.append(doc)
                print(f"   ✓ {file_path.name}: 提取了{len(text)}字符")
        return docs

    def _split_documents(self, documents: List[Document]) -> List[Document]:
        """智能分块（带元数据增强）"""
        all_chunks = []
        for idx, doc in enumerate(documents):
            # 判断文档类型
            source = doc.metadata.get("source", "")
            is_markdown = source.endswith((".md", ".markdown"))
            is_code = source.endswith((".py", ".js", ".java"))

            if is_markdown:
                # Markdown分块
                from langchain_text_splitters import MarkdownHeaderTextSplitter
                md_splitter = MarkdownHeaderTextSplitter(
                    headers_to_split_on=[("#", "Header1"), ("##", "Header2")]
                )
                chunks = md_splitter.split_text(doc.page_content)
            elif is_code:
                # 代码分块
                from langchain_text_splitters import RecursiveCharacterTextSplitter, Language
                ext = source.split(".")[-1]
                language_map = {"py": Language.PYTHON, "js": Language.JS, "java": Language.JAVA}
                code_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language_map.get(ext, Language.PYTHON),
                    chunk_size=800,
                    chunk_overlap=0
                )
                chunks = code_splitter.split_documents([doc])
            else:
                # 默认分块
                chunks = self.text_splitter.split_documents([doc])

            # 增强元数据
            for i, chunk in enumerate(chunks):
                chunk.metadata.update({
                    "parent_id": f"doc_{idx}",
                    "splitter": "markdown" if is_markdown else "code" if is_code else "fallback",
                    "chunk_idx": i,
                })

            all_chunks.extend(chunks)

        print(f"✂️  分块完成：{len(all_chunks)}个文本块")
        return all_chunks

    async def process(self, texts: List[str], files: Optional[List[Union[Path, str]]] = None,
                      metadatas: Optional[List[Dict]] = None) -> Any:
        """完整处理流程：加载 → 分块 → 向量化"""
        print("📄 加载文档...")

        # 1. 处理纯文本
        documents = self._load_text_documents(texts, metadatas or [])

        # 2. 处理文件（OCR）
        if files:
            file_docs = await self._load_file_documents([Path(f) for f in files])
            documents.extend(file_docs)

        print(f"   总计 {len(documents)} 个文档")

        # 3. 智能分块
        chunks = self._split_documents(documents)
        print(f"   生成了 {len(chunks)} 个文本块")

        # 4. 创建向量存储
        print("🔢 创建向量存储...")

        # 初始化 vector_manager
        if self.vector_manager is None:
            self.vector_manager = VectorStoreManager(self.config)

        vector_store = self.vector_manager.create_from_documents(chunks)
        print("   向量存储创建完成")

        self.vector_store = vector_store

        return vector_store

    # ==================== 新增：生产级OCR文档添加方法 ====================
    async def add_ocr_documents(self, files: List[Union[Path, str]]) -> List[Document]:
        """生产级：动态添加OCR文档到向量库"""
        if not self.ocr_processor.is_available():
            print("⚠️  OCR功能不可用，无法添加OCR文档")
            return []

        if not self.vector_manager or not self.vector_store:
            raise ValueError("向量存储未初始化，请先调用 process()")

        print(f"📷 动态添加OCR文档：{len(files)}个文件")

        ocr_docs = []
        for file_path in files:
            text = await self.ocr_processor.extract_text(Path(file_path))
            if text:
                doc = self.ocr_processor.create_document(Path(file_path), text)
                # 修改metadata标记为OCR文档
                doc.metadata.update({
                    "source": f"ocr_{Path(file_path).name}",
                    "doc_type": "ocr_document",
                    "added_at": time.time()
                })
                ocr_docs.append(doc)
                print(f"   ✓ {Path(file_path).name}: 提取了{len(text)}字符")

        if ocr_docs:
            # 添加到向量库
            self.vector_manager.add_documents(ocr_docs)

            # 更新检索器（必须重新初始化BM25）
            if self.retriever:
                self.retriever.update_documents(ocr_docs)

            print(f"✅ 成功添加 {len(ocr_docs)} 个OCR文档")

        return ocr_docs
    # ==================== 新增结束 ====================