import gradio as gr
import requests
import json

API_URL = "http://localhost:8000/api"


class SelfRAGWebUI:
    def __init__(self, api_url: str = "http://localhost:8000/api"):
        self.api_url = api_url

    def query(self, question: str, stream: bool) -> tuple:
        if not question.strip():
            return "请输入问题", "0.00", "0", "[]", "等待查询...", True, ""

        try:
            # 统一调用：只请求一次API
            response = requests.post(
                f"{self.api_url}/query",
                json={"question": question},
                timeout=30
            )
            result = response.json()

            # 流式模式：额外获取流式文本
            answer = result["answer"]
            if stream:
                stream_resp = requests.post(
                    f"{self.api_url}/query?stream=true",
                    json={"question": question},
                    stream=True, timeout=60
                )
                full_text = ""
                for line in stream_resp.iter_lines():
                    if line and b"data: " in line:
                        chunk = json.loads(line.replace(b"data: ", b"").decode())
                        if '"done"' in line.decode():
                            break
                        full_text += chunk.get("text", "")
                answer = full_text

            # 统一格式化输出
            review_msg = ""
            if result.get("needs_review"):
                review_msg = f"需要人工审核（任务ID: {result.get('review_task_id', 'N/A')}）"

            return (
                answer,
                f"{result['confidence']:.2f}",
                str(result["iteration"]),
                json.dumps(result["sources"][:3]),
                "查询完成",
                True,
                review_msg
            )
        except Exception as e:
            return f"查询失败: {str(e)}", "0.00", "0", "[]", "错误", True, ""

    def clear_all(self):
        """清空所有组件"""
        return "", "0.00", "0", "[]", "等待查询...", True, ""


def create_interface():
    """创建Gradio界面"""
    ui = SelfRAGWebUI()

    with gr.Blocks(title="Self-RAG 智能问答", theme=gr.themes.Soft()) as demo:
        # 标题区
        gr.Markdown("Self-RAG 智能问答系统")

        # 输入区
        with gr.Row():
            question_input = gr.Textbox(
                label="输入问题",
                placeholder="请输入您的问题...",
                lines=3,
                max_lines=5
            )

        # 控制区
        with gr.Row():
            stream_checkbox = gr.Checkbox(label="启用流式输出", value=True)
            submit_btn = gr.Button("提交查询", variant="primary", scale=2)
            clear_btn = gr.Button("清空", variant="secondary")

        # 答案展示区
        answer_output = gr.Markdown(label="回答内容", value="等待查询...")

        # 状态区
        with gr.Row():
            confidence_display = gr.Textbox(label="置信度", value="0.00", interactive=False, scale=1)
            iteration_display = gr.Textbox(label="迭代次数", value="0", interactive=False, scale=1)

        # 来源追溯区
        sources_display = gr.JSON(label="检索来源（前3条）", value=[])

        # 状态与提示
        status_display = gr.Textbox(label="状态", value="等待查询...", interactive=False)
        review_display = gr.Markdown(label="人工审核提示", visible=True)

        # 示例问题
        gr.Examples(
            examples=[
                ["什么是Self-RAG？"],
                ["向量数据库如何选型？"],
                ["LangGraph有什么优势？"]
            ],
            inputs=[question_input],
            label="示例问题"
        )

        # 事件绑定
        submit_btn.click(
            fn=ui.query,
            inputs=[question_input, stream_checkbox],
            outputs=[
                answer_output,
                confidence_display,
                iteration_display,
                sources_display,
                status_display,
                submit_btn,
                review_display
            ]
        )

        clear_btn.click(
            fn=ui.clear_all,
            outputs=[
                question_input,
                answer_output,
                confidence_display,
                iteration_display,
                sources_display,
                status_display,
                submit_btn,
                review_display
            ]
        )

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        debug=False,
        show_error=True
    )