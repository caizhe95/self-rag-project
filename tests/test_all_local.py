# test_all_local.py
import sys
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any

# 添加到项目路径
sys.path.append(str(Path(__file__).parent))

from core.rag_chain import SelfRAGChain
from config.setting import RAGConfig
from core.ocr_processor import OCRProcessor


class LocalTestSuite:
    """本地完整测试套件"""

    def __init__(self):
        self.config = RAGConfig()
        self.rag = SelfRAGChain(self.config)
        self.results: List[Dict[str, Any]] = []

    async def run_all_tests(self):
        """运行所有测试"""
        print("=" * 70)
        print("🚀 开始本地完整测试")
        print("=" * 70)

        # 测试1：基础OCR功能
        print("\n【测试1】OCR基础功能")
        result1 = await self._test_ocr_basic()
        self.results.append(result1)

        # 测试2：文档索引
        print("\n【测试2】文档索引")
        result2 = await self._test_indexing()
        self.results.append(result2)

        # 测试3：基本RAG查询
        print("\n【测试3】基本RAG查询")
        result3 = await self._test_basic_rag()
        self.results.append(result3)

        # 测试4：Self-RAG迭代
        print("\n【测试4】Self-RAG迭代优化")
        result4 = await self._test_self_rag_iteration()
        self.results.append(result4)

        # 测试5：人机协作
        print("\n【测试5】人机协作流程")
        result5 = await self._test_human_review()
        self.results.append(result5)

        # 测试6：OCR集成
        print("\n【测试6】OCR + RAG集成")
        result6 = await self._test_ocr_integration()
        self.results.append(result6)

        return self._generate_report()

    async def _test_ocr_basic(self) -> Dict[str, Any]:
        """测试OCR基础功能"""
        ocr = OCRProcessor(language="chi_sim+eng", enabled=True)

        # 创建测试图片
        from PIL import Image, ImageDraw

        test_image = Path("./data/test_local.png")
        test_image.parent.mkdir(exist_ok=True)

        img = Image.new('RGB', (300, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "PyCharm测试123", fill='black')
        draw.text((10, 50), "Local OCR ABC", fill='blue')
        img.save(test_image)

        # 测试识别
        text = await ocr.extract_text(test_image)

        passed = text is not None and len(text) > 0

        return {
            "test_name": "OCR基础功能",
            "passed": passed,
            "message": "OCR识别成功" if passed else "OCR识别失败"
        }

    async def _test_indexing(self) -> Dict[str, Any]:
        """测试文档索引"""
        sample_docs = [
            {"text": "LangChain是LLM框架", "metadata": {"source": "test1"}},
            {"text": "Self-RAG是增强版RAG", "metadata": {"source": "test2"}}
        ]

        try:
            await self.rag.aindex_documents(
                texts=[doc["text"] for doc in sample_docs],
                metadatas=[doc["metadata"] for doc in sample_docs]
            )
            return {
                "test_name": "文档索引",
                "passed": True,
                "message": "索引成功"
            }
        except Exception as e:
            return {
                "test_name": "文档索引",
                "passed": False,
                "message": f"索引失败: {e}"
            }

    async def _test_basic_rag(self) -> Dict[str, Any]:
        """测试基本RAG查询"""
        try:
            result = await self.rag.query("什么是LangChain？")

            passed = (
                    len(result["answer"]) > 50 and
                    result["confidence"] > 0.3
            )

            return {
                "test_name": "基本RAG查询",
                "passed": passed,
                "message": f"回答长度{len(result['answer'])}, 置信度{result['confidence']:.2f}"
            }
        except Exception as e:
            return {
                "test_name": "基本RAG查询",
                "passed": False,
                "message": f"查询失败: {e}"
            }

    async def _test_self_rag_iteration(self) -> Dict[str, Any]:
        """测试Self-RAG迭代"""
        result = await self.rag.query("解释量子计算和经典计算的区别")

        passed = result["iteration"] >= 1

        return {
            "test_name": "Self-RAG迭代",
            "passed": passed,
            "message": f"查询迭代了{result['iteration']}次"
        }

    async def _test_human_review(self) -> Dict[str, Any]:
        """测试人机协作触发"""
        # 问一个模糊问题，触发审核
        result = await self.rag.query("LangGraph能用来做游戏吗？")

        passed = "review_task_id" in result

        return {
            "test_name": "人机协作触发",
            "passed": passed,
            "message": "触发人工审核" if passed else "未触发审核"
        }

    async def _test_ocr_integration(self) -> Dict[str, Any]:
        """测试OCR + RAG集成"""
        # 创建测试图片
        from PIL import Image, ImageDraw

        test_image = Path("./data/test_ocr_rag.png")
        img = Image.new('RGB', (400, 100), color='white')
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "LangGraph支持人机协作", fill='black')
        draw.text((10, 50), "Human-in-the-loop feature", fill='blue')
        img.save(test_image)

        # 查询
        result = await self.rag.query(
            "图片中提到的功能是什么？",
            files=[str(test_image)]
        )

        passed = "人机协作" in result["answer"] or "human" in result["answer"].lower()

        return {
            "test_name": "OCR + RAG集成",
            "passed": passed,
            "message": "OCR识别并用于RAG回答" if passed else "OCR集成失败"
        }

    def _generate_report(self) -> Dict[str, Any]:
        """生成测试报告"""
        passed = sum(1 for r in self.results if r["passed"])
        total = len(self.results)

        report = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "summary": {
                "passed": f"{passed}/{total}",
                "success_rate": f"{(passed / total * 100):.1f}%" if total > 0 else "0%"
            },
            "details": self.results
        }

        print("\n" + "=" * 70)
        print("📊 本地测试报告")
        print("=" * 70)
        for result in self.results:
            status = "✅ 通过" if result["passed"] else "❌ 失败"
            print(f"{status} {result['test_name']}: {result['message']}")

        print("\n" + f"总计: {report['summary']['passed']}, 成功率: {report['summary']['success_rate']}")
        print("=" * 70)

        return report


if __name__ == "__main__":
    # 确保测试模式
    import os

    os.environ["LOCAL_TEST"] = "true"
    os.environ["HUMAN_REVIEW_ENABLED"] = "true"

    # 运行测试
    tester = LocalTestSuite()
    asyncio.run(tester.run_all_tests())