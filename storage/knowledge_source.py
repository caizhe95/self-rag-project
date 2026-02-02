from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from pathlib import Path
import aiofiles
import json
import hashlib
from datetime import datetime
import os


class KnowledgeSource(ABC):
    """知识源抽象基类"""

    def __init__(self, source_id: str, priority: int = 100):
        self.source_id = source_id
        self.priority = priority
        self.last_sync: Optional[datetime] = None

    @abstractmethod
    async def load_documents(self) -> List[Document]:
        """异步加载文档"""
        pass

    def _create_document(self, content: str, metadata: Dict[str, Any]) -> Document:
        """统一创建Document"""
        meta = {
            "source_id": self.source_id,
            "source_type": self.__class__.__name__,
            "imported_at": datetime.now().isoformat(),
            **metadata
        }
        meta["content_hash"] = hashlib.md5(content.encode()).hexdigest()[:16]
        return Document(page_content=content, metadata=meta)


class FileSystemSource(KnowledgeSource):
    """批量导入：./data/knowledge/ 目录下的文件"""

    def __init__(self, path: str = "./data/knowledge", priority: int = 10):
        super().__init__(source_id="filesystem", priority=priority)
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    async def load_documents(self) -> List[Document]:
        """支持 MD, TXT, JSON"""
        docs = []
        files = list(self.path.glob("*.md")) + \
                list(self.path.glob("*.txt")) + \
                list(self.path.glob("*.json"))

        for file_path in files:
            try:
                if file_path.suffix == '.json':
                    docs.extend(await self._load_json(file_path))
                else:
                    docs.extend(await self._load_text(file_path))
            except Exception as e:
                print(f"[{self.source_id}] 加载失败 {file_path}: {e}")

        print(f"[{self.source_id}] 加载了 {len(docs)} 个文档")
        self.last_sync = datetime.now()
        return docs

    async def _load_text(self, file_path: Path) -> List[Document]:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            content = await f.read()

        return [self._create_document(
            content=content,
            metadata={
                "source": file_path.name,
                "file_type": file_path.suffix
            }
        )]

    async def _load_json(self, file_path: Path) -> List[Document]:
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            data = json.loads(await f.read())

        docs = []
        if isinstance(data, list):
            for idx, item in enumerate(data):
                if isinstance(item, dict) and "content" in item:
                    docs.append(self._create_document(
                        content=item["content"],
                        metadata={
                            "source": file_path.name,
                            "entry_id": idx,
                            **item.get("metadata", {})
                        }
                    ))
        return docs


class UploadSource(KnowledgeSource):
    """处理用户上传的文档"""

    def __init__(self, upload_path: str = "./data/uploads", priority: int = 5):
        super().__init__(source_id="user_upload", priority=priority)
        self.upload_path = Path(upload_path)
        self.upload_path.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, filename: str, content: bytes, metadata: Dict[str, Any]) -> str:
        """保存上传文件"""
        file_path = self.upload_path / filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

        meta_path = self.upload_path / f"{filename}.meta.json"
        async with aiofiles.open(meta_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps({
                "uploaded_at": datetime.now().isoformat(),
                **metadata
            }, ensure_ascii=False, indent=2))

        return str(file_path)

    async def load_documents(self) -> List[Document]:
        """加载所有上传的文件"""
        docs = []
        files = list(self.upload_path.iterdir())

        for file_path in files:
            if file_path.suffix == '.meta.json':
                continue

            meta_path = self.upload_path / f"{file_path.name}.meta.json"
            metadata = {}

            if meta_path.exists():
                async with aiofiles.open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.loads(await f.read())

            try:
                async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                    content = await f.read()

                docs.append(self._create_document(
                    content=content,
                    metadata={
                        "source": f"upload_{file_path.name}",
                        **metadata
                    }
                ))
            except Exception as e:
                print(f"[{self.source_id}] 加载失败 {file_path}: {e}")

        print(f"[{self.source_id}] 加载了 {len(docs)} 个上传文档")
        self.last_sync = datetime.now()
        return docs


class KnowledgeManager:
    """统一管理所有知识源"""

    def __init__(self, sources: List[KnowledgeSource]):
        self.sources = sorted(sources, key=lambda s: s.priority)

    async def load_all_documents(self, deduplicate: bool = True) -> List[Document]:
        """从所有源加载文档"""
        all_docs = []
        seen_hashes = set()

        for source in self.sources:
            try:
                docs = await source.load_documents()

                for doc in docs:
                    content_hash = doc.metadata.get("content_hash")

                    if deduplicate and content_hash in seen_hashes:
                        continue

                    seen_hashes.add(content_hash)
                    all_docs.append(doc)

            except Exception as e:
                print(f"[KnowledgeManager] {source.source_id} 加载失败: {e}")
                continue

        print(f"[KnowledgeManager] 总计加载 {len(all_docs)} 个唯一文档")
        return all_docs

    def add_source(self, source: KnowledgeSource):
        self.sources.append(source)
        self.sources.sort(key=lambda s: s.priority)