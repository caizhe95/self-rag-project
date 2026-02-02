# webui_user.py - 用户查询界面（简洁版）
import os
import requests
import gradio as gr
from typing import List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class UserRAGInterface:
    """用户端界面：专注查询体验"""

    def __init__(self, api_url: str = None):
        self.api_url = api_url or API_URL
        self.session_id = None
        self.chat_history: List[Dict[str, str]] = []

    def query(self, question: str, file_paths: List[str]) -> tuple:
        """执行查询"""
        if not question.strip():
            return "请输入问题", "", ""

        # 处理文件上传（OCR）
        files = file_paths if file_paths else None

        try:
            payload = {
                "question": question,
                "session_id": self.session_id,
                "chat_history": self.chat_history[-6:],  # 保留最近3轮
                "files": files
            }

            response = requests.post(
                f"{self.api_url}/api/query",
                json=payload,
                timeout=120
            )

            if response.status_code == 200:
                data = response.json().get("data", {})

                # 更新历史
                self.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": data.get("answer", "")}
                ])

                # 格式化输出
                answer = data.get("answer", "无回答")
                confidence = data.get("confidence", 0)
                iteration = data.get("iteration", 0)
                sources = data.get("sources", [])
                review_status = data.get("review_status")

                # 置信度指示器
                confidence_emoji = "🟢" if confidence >= 0.7 else "🟡" if confidence >= 0.5 else "🔴"

                # 构建元信息
                meta_info = f"{confidence_emoji} 置信度: {confidence:.0%} | 迭代优化: {iteration}次"

                if review_status:
                    meta_info += f" | ⚠️ 已提交人工审核 [{data.get('review_task_id', '')[:6]}]"

                # 格式化来源
                if sources:
                    sources_text = "docs: " + " | ".join([
                        f"[^{i + 1}^] {s['source']}: {s['content_preview'][:30]}..."
                        for i, s in enumerate(sources[:3])
                    ])
                else:
                    sources_text = "docs: 无"

                return answer, meta_info, sources_text

            else:
                return f"请求失败: {response.status_code}", "", ""

        except requests.exceptions.Timeout:
            return "⏱️ 请求超时，请稍后重试", "", ""
        except Exception as e:
            return f"❌ 错误: {str(e)}", "", ""

    def clear_history(self):
        """清空对话历史"""
        self.chat_history = []
        self.session_id = None
        return [], "历史已清空"

    def create_interface(self):
        """创建用户界面"""
        with gr.Blocks(title="Self-RAG 智能问答系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🤖 Self-RAG 智能问答系统
            基于检索增强生成技术，支持文档理解、迭代优化和人工审核机制
            """)

            with gr.Row():
                # 左侧：输入区
                with gr.Column(scale=1):
                    gr.Markdown("### 📝 输入问题")

                    question_input = gr.Textbox(
                        label="您的问题",
                        placeholder="例如：什么是Self-RAG？",
                        lines=3
                    )

                    file_upload = gr.File(
                        label="上传文档/图片（支持OCR）",
                        file_count="multiple",
                        file_types=[".pdf", ".png", ".jpg", ".jpeg"]
                    )

                    with gr.Row():
                        submit_btn = gr.Button("🔍 查询", variant="primary", scale=3)
                        clear_btn = gr.Button("🗑️ 清空历史", variant="secondary", scale=1)

                    # 系统状态指示器
                    status_text = gr.Textbox(
                        label="系统状态",
                        value="✅ 系统就绪",
                        interactive=False
                    )

                # 右侧：结果区
                with gr.Column(scale=2):
                    gr.Markdown("### 💡 回答")
                    answer_output = gr.Markdown(label="回答")

                    meta_output = gr.Textbox(
                        label="评估信息",
                        interactive=False,
                        value=""
                    )

                    with gr.Accordion("📚 参考来源", open=False):
                        sources_output = gr.Markdown(label="来源")

            # 事件绑定
            submit_btn.click(
                fn=self.query,
                inputs=[question_input, file_upload],
                outputs=[answer_output, meta_output, sources_output]
            ).then(
                fn=lambda: "✅ 查询完成",
                outputs=[status_text]
            )

            clear_btn.click(
                fn=self.clear_history,
                outputs=[answer_output, status_text]
            )

            # 页脚信息
            gr.Markdown("---")
            gr.Markdown("💡 提示：回答置信度较低时，系统会自动触发人工审核机制")

        return demo


if __name__ == "__main__":
    # 检查后端连接
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ 已连接到后端服务")
            data = response.json()
            print(f"   模式: {data.get('mode', 'unknown')}")
            print(f"   模型: {data.get('model', 'unknown')}")
            print(f"   文档数: {data.get('document_count', 0)}")
        else:
            print(f"⚠️ 后端响应异常: {response.status_code}")
    except Exception as e:
        print(f"❌ 无法连接到后端: {e}")
        print("请确保 server.py 正在运行")

    # 启动界面
    ui = UserRAGInterface()
    demo = ui.create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,  # 用户界面用 7860
        share=False,
        show_error=True
    )