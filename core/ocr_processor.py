# core/ocr_processor.py
from pathlib import Path
from typing import Optional, Dict, Any
from langchain_core.documents import Document
import os
import pytesseract

# 读取自定义路径
tesseract_cmd = os.getenv("TESSERACT_CMD")
if tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
    print(f"✅ 已设置Tesseract路径: {tesseract_cmd}")


class OCRProcessor:
    """OCR处理器：图片/PDF → 可检索文本（面试级）"""

    def __init__(self, language: str = "chi_sim+eng", enabled: bool = True):
        """
        初始化 OCR 处理器

        Args:
            language: Tesseract 语言包
            enabled: 是否启用 OCR
        """
        self.language = language
        self.enabled = enabled
        self._available = False

        if not enabled:
            print("⚠️  OCR功能已手动禁用")
            return

        try:
            import pytesseract
            pytesseract.get_tesseract_version()
            self._available = True
            print("✅ OCR功能已启用")
        except ImportError:
            print("⚠️  未安装pytesseract，OCR功能将不可用")
            print("   安装: pip install pytesseract pillow pdfplumber")
        except Exception as e:
            print(f"⚠️  Tesseract未安装，OCR功能将不可用: {e}")
            print("   Ubuntu安装: sudo apt-get install tesseract-ocr-chi-sim")

    def is_available(self) -> bool:
        """检查 OCR 是否可用"""
        return self._available

    async def extract_text(self, file_path: Path) -> Optional[str]:
        """异步提取文本"""
        if not self._available:
            return None

        import pytesseract
        from PIL import Image
        import pdfplumber

        try:
            if file_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".webp"]:
                # 图片OCR
                img = Image.open(file_path)
                return pytesseract.image_to_string(img, lang=self.language)

            elif file_path.suffix.lower() == ".pdf":
                # PDF文本提取（优先使用pdfplumber）
                with pdfplumber.open(file_path) as pdf:
                    texts = []
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            texts.append(f"[Page {i + 1}]\n{text}")
                    return "\n\n".join(texts) if texts else None

        except Exception as e:
            print(f"❌ OCR提取失败 {file_path}: {e}")
            return None

    def create_document(self, file_path: Path, text: str) -> Document:
        """创建OCR文档"""
        return Document(
            page_content=text,
            metadata={
                "source": file_path.name,
                "type": "ocr_extracted",
                "original_path": str(file_path),
                "file_type": file_path.suffix.lower()
            }
        )