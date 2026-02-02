# main_local.py - 本地小模型专用（llama3.2:3b）
import asyncio
import os
import sys
import requests
from dotenv import load_dotenv

if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

os.environ["NO_PROXY"] = "127.0.0.1,localhost"
load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")


class SimpleSessionManager:
    """轻量级Session管理（小模型资源受限）"""

    def __init__(self):
        self.history = []
        self.last_question = ""
        self.turn_count = 0

    def should_new_session(self, question: str) -> bool:
        """关键词匹配检测话题（快速不耗资源）"""
        if not self.last_question or self.turn_count >= 3:  # 最多3轮
            return self.turn_count >= 3

        # 简单关键词重叠
        last = set(self.last_question.lower().split())
        curr = set(question.lower().split())
        overlap = len(last & curr) / len(last) if last else 1.0

        return overlap < 0.3  # 30%以下视为新话题

    def get_session(self, question: str) -> tuple:
        is_new = self.should_new_session(question)
        if is_new:
            self.history = []
            self.turn_count = 0
            print("🆕 新话题")
        else:
            print(f"💬 继续对话(第{self.turn_count + 1}轮)")

        self.last_question = question
        return f"cli_{os.getpid()}_{self.turn_count}", self.history.copy(), is_new

    def update(self, q: str, a: str):
        self.history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": a}])
        self.turn_count += 1
        # 只保留最近2轮（小模型上下文有限）
        if len(self.history) > 4:
            self.history = self.history[-4:]


def show_result(data):
    confidence = data.get('confidence', 0)
    emoji = "🟢" if confidence > 0.7 else "🟡" if confidence > 0.5 else "🔴"
    print(f"\n答案: {data.get('answer', '无回答')}")
    print(f"{emoji} 置信度: {confidence:.0%} | {data.get('iteration', 0)}次迭代", end="")
    print(f" | ⚠️审核" if data.get('review_task_id') else " | ✓通过")


async def main():
    print(f"Self-RAG Client [本地模式|llama3.2:3b]")
    print(f"📝 特点: 关键词话题检测 | 最多3轮对话 | 轻快省资源")
    print(f"💡 输入 'exit'退出 | 不同话题自动换Session\n")

    mgr = SimpleSessionManager()

    while True:
        try:
            q = input("问题: ").strip()
            if q in ['exit', 'quit']:
                break
            if not q:
                continue

            sid, history, _ = mgr.get_session(q)
            print("思考...", end="", flush=True)

            res = requests.post(
                f"{API_URL}/api/query",
                json={"question": q, "session_id": sid, "chat_history": history},
                timeout=120,
                verify=False
            )

            data = res.json().get("data", {})
            show_result(data)
            mgr.update(q, data.get('answer', ''))

        except requests.exceptions.ConnectionError:
            print("\n❌ Server未启动")
        except Exception as e:
            print(f"\n❌ 错误: {e}")


if __name__ == "__main__":
    asyncio.run(main())