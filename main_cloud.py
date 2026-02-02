# main_cloud.py - 云端大模型专用（deepseek:32b）
import asyncio
import os
import sys
import uuid
import requests
import numpy as np
from dotenv import load_dotenv

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")
OLLAMA_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")


class DeepSeekSessionManager:
    """DeepSeek-32B专用Session（语义理解强，支持长上下文）"""

    def __init__(self):
        self.history = []
        self.last_emb = None
        self.turns = 0
        self.current_sid = f"ds_{uuid.uuid4().hex[:6]}"

    def get_embedding(self, text: str) -> np.ndarray:
        """调用本地bge-m3获取embedding（DeepSeek配bge-m3中文好）"""
        try:
            r = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": "bge-m3:latest", "prompt": text[:256]},
                timeout=5
            )
            if r.status_code == 200:
                return np.array(r.json().get("embedding", []))
        except:
            pass
        return np.array([])

    def similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.size == 0 or v2.size == 0:
            return 1.0  # 失败时保守处理，不切换
        v1, v2 = v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)
        return float(np.dot(v1, v2))

    def should_new_session(self, q: str) -> bool:
        """语义相似度检测（32B理解力强，阈值设0.65）"""
        if self.turns == 0:
            return False
        if self.turns >= 10:  # 32B支持长上下文，但10轮后重置保性能
            print(f"（已{self.turns}轮，重置会话）")
            return True

        curr_emb = self.get_embedding(q)
        if curr_emb.size > 0 and self.last_emb is not None:
            sim = self.similarity(curr_emb, self.last_emb)
            print(f" [相似度{sim:.0%}]", end="")
            if sim < 0.65:  # 65%阈值（比小模型严格，因embedding准）
                print(" 话题切换")
                return True
        return False

    def get_session(self, q: str) -> tuple:
        if self.should_new_session(q):
            self.current_sid = f"ds_{uuid.uuid4().hex[:6]}"
            self.history = []
            self.turns = 0
            print(f"🆕 新会话[{self.current_sid}]")
        else:
            print(f"💬 继续[{self.current_sid[:6]}]第{self.turns + 1}轮")

        self.last_emb = self.get_embedding(q)
        return self.current_sid, self.history.copy()

    def update(self, q: str, a: str):
        self.history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
        self.turns += 1


def show_result(data):
    c = data.get('confidence', 0)
    emoji = "🟢" if c >= 0.75 else "🟡" if c >= 0.5 else "🔴"
    print(f"\n📝 {data.get('answer', '无')[:400]}{'...' if len(data.get('answer', '')) > 400 else ''}")
    print(f"{emoji} {c:.0%}置信 | {data.get('iteration', 0)}轮Self-RAG", end="")
    if data.get('review_task_id'):
        print(f" | 🔍审核ID:{data['review_task_id'][:6]}")
    else:
        print(" | ✓通过")


async def main():
    print(f"🚀 Self-RAG Client [云端模式|DeepSeek-32B]")
    print(f"🧠 嵌入模型: bge-m3 | 语义相似度阈值: 65% | 最长10轮")
    print(f"⏱️  超时: 120秒（32B推理较慢）")
    print(f"💡 'exit'=退出\n")

    # 检查模型
    try:
        r = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
        if r.status_code == 200:
            models = [m['name'] for m in r.json().get('models', [])]
            ds = [m for m in models if 'deepseek' in m]
            if ds:
                print(f"✅ 检测到DeepSeek: {ds[0]}\n")
            else:
                print(f"⚠️ 未检测到DeepSeek，当前: {models[:2]}\n")
    except:
        print("⚠️ Ollama连接失败，embedding检测将失效\n")

    mgr = DeepSeekSessionManager()

    while True:
        try:
            q = input("问题: ").strip()
            if q in ['exit', 'quit']:
                break
            if not q:
                continue

            sid, hist = mgr.get_session(q)
            print("推理中...", end="", flush=True)

            res = requests.post(
                f"{API_URL}/api/query",
                json={"question": q, "session_id": sid, "chat_history": hist},
                timeout=120,  # 32B慢，给120秒
                verify=False
            )

            data = res.json().get("data", {})
            show_result(data)
            mgr.update(q, data.get('answer', ''))

        except requests.exceptions.Timeout:
            print("\n⏱️ 超时（32B正常，可重试）")
        except Exception as e:
            print(f"\n❌ {e}")


if __name__ == "__main__":
    asyncio.run(main())