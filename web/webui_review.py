# webui_review.py - 人工审核界面（专业版）
import os
import requests
import gradio as gr
from typing import List, Dict, Any
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class ReviewerInterface:
    """审核员界面：专业审核工作台"""

    def __init__(self, api_url: str = None):
        self.api_url = api_url or API_URL
        self.current_task_id = None
        self.stats = {"approved": 0, "rejected": 0, "modified": 0}

    def load_pending_list(self) -> List[List[str]]:
        """加载待审核任务列表"""
        try:
            response = requests.get(f"{self.api_url}/api/reviews/pending", timeout=10)
            if response.status_code == 200:
                data = response.json()
                tasks = data.get("reviews", [])

                # 格式化表格数据
                rows = []
                for task in tasks:
                    created_time = datetime.fromtimestamp(task["created_at"]).strftime("%H:%M:%S")
                    rows.append([
                        task["task_id"][:8],
                        task["query"][:40] + "..." if len(task["query"]) > 40 else task["query"],
                        f"{task['confidence']:.0%}",
                        f"{task['hallucination_risk']:.0%}",
                        task["trigger_reason"][:25] + "..." if len(task["trigger_reason"]) > 25 else task[
                            "trigger_reason"],
                        created_time
                    ])
                return rows
            return []
        except Exception as e:
            print(f"加载待审核列表失败: {e}")
            return []

    def load_task_detail(self, task_id: str) -> tuple:
        """加载任务详情"""
        if not task_id:
            return tuple([gr.update(visible=False)] * 7 + ["请选择任务"])  # 隐藏所有详情区

        try:
            response = requests.get(f"{self.api_url}/api/reviews/{task_id}", timeout=10)
            if response.status_code == 200:
                data = response.json().get("review", {})
                self.current_task_id = task_id

                # 指标数据
                metrics = data.get("metrics", {})
                confidence = metrics.get("confidence", 0)
                hallucination = metrics.get("hallucination_risk", 0)
                relevance = metrics.get("retrieval_relevance", 0)

                # 格式化文档
                docs = data.get("documents", [])
                docs_text = "\n\n---\n\n".join([
                    f"**[{i + 1}] {doc['source']}**\n{doc['content'][:500]}{'...' if len(doc['content']) > 500 else ''}"
                    for i, doc in enumerate(docs[:5])  # 最多显示5篇
                ])

                # 风险提示
                risk_alerts = []
                if confidence < 0.5:
                    risk_alerts.append("⚠️ 置信度低于50%")
                if hallucination > 0.5:
                    risk_alerts.append("⚠️ 幻觉风险较高")
                if relevance < 0.3:
                    risk_alerts.append("⚠️ 检索相关性低")

                risk_text = " | ".join(risk_alerts) if risk_alerts else "✅ 风险指标正常"

                return tuple([
                    gr.update(visible=True),  # 详情区显示
                    data.get("query", ""),
                    data.get("original_answer", ""),
                    docs_text,
                    f"{confidence:.0%}",
                    f"{hallucination:.0%}",
                    f"{relevance:.0%}",
                    risk_text
                ])
            else:
                return tuple([gr.update(visible=False)] * 7 + ["加载失败"])
        except Exception as e:
            return tuple([gr.update(visible=False)] * 7 + [f"错误: {str(e)}"])

    def submit_review_action(self, action: str, modified_text: str, comment: str, reviewer_name: str) -> tuple:
        """提交审核结果"""
        if not self.current_task_id:
            return "❌ 请先选择审核任务", self.get_stats_text()

        if action == "modified" and not modified_text.strip():
            return "❌ 修改模式必须填写修改后的答案", self.get_stats_text()

        try:
            payload = {
                "task_id": self.current_task_id,
                "action": action,
                "modified_answer": modified_text if action == "modified" else None,
                "comment": comment,
                "reviewer": reviewer_name or "anonymous"
            }

            response = requests.post(
                f"{self.api_url}/api/reviews/submit",
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                result = response.json()
                if result.get("success"):
                    # 更新统计
                    self.stats[action] = self.stats.get(action, 0) + 1

                    task_id_short = self.current_task_id[:8]
                    self.current_task_id = None

                    return f"✅ 审核已提交 [{task_id_short}]: {action}", self.get_stats_text()
                else:
                    return f"❌ 提交失败: {result.get('message', '未知错误')}", self.get_stats_text()
            else:
                return f"❌ API错误: {response.status_code}", self.get_stats_text()

        except Exception as e:
            return f"❌ 提交异常: {str(e)}", self.get_stats_text()

    def get_stats_text(self) -> str:
        """获取统计文本"""
        total = sum(self.stats.values())
        return f"今日审核: 通过 {self.stats['approved']} | 拒绝 {self.stats['rejected']} | 修改 {self.stats['modified']} | 总计 {total}"

    def create_interface(self):
        """创建审核界面"""
        with gr.Blocks(title="Self-RAG 人工审核系统", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 🔍 Self-RAG 人工审核工作台
            审核低置信度或高幻觉风险的AI生成答案，确保输出质量
            """)

            # 统计栏
            stats_text = gr.Textbox(
                value=self.get_stats_text(),
                label="审核统计",
                interactive=False
            )

            with gr.Row():
                # 左侧：任务列表
                with gr.Column(scale=1):
                    gr.Markdown("### 📋 待审核任务")

                    refresh_btn = gr.Button("🔄 刷新列表", variant="secondary")

                    task_table = gr.Dataframe(
                        headers=["任务ID", "问题摘要", "置信度", "幻觉风险", "触发原因", "时间"],
                        datatype="str",
                        interactive=False,
                        row_count=8,
                        wrap=True
                    )

                    selected_task = gr.Textbox(
                        label="当前选中任务",
                        interactive=False,
                        value=""
                    )

                # 右侧：审核工作区
                with gr.Column(scale=2):
                    gr.Markdown("### 🔎 任务详情")

                    with gr.Group(visible=False) as detail_group:
                        # 风险提示
                        risk_alert = gr.Textbox(
                            label="⚠️ 风险提示",
                            interactive=False,
                            value=""
                        )

                        with gr.Row():
                            query_text = gr.Textbox(
                                label="用户问题",
                                lines=2,
                                interactive=False
                            )

                            with gr.Column():
                                conf_score = gr.Textbox(label="置信度", interactive=False)
                                hall_score = gr.Textbox(label="幻觉风险", interactive=False)
                                rel_score = gr.Textbox(label="检索相关性", interactive=False)

                        answer_text = gr.Textbox(
                            label="AI生成答案（待审核）",
                            lines=6,
                            interactive=False
                        )

                        reference_docs = gr.Textbox(
                            label="参考文档",
                            lines=4,
                            interactive=False
                        )

                        # 审核操作区
                        gr.Markdown("### ✅ 审核操作")

                        with gr.Row():
                            with gr.Column(scale=1):
                                action_radio = gr.Radio(
                                    choices=[
                                        ("✅ 通过（直接采纳）", "approved"),
                                        ("❌ 拒绝（重新生成）", "rejected"),
                                        ("✏️ 修改（人工修正）", "modified")
                                    ],
                                    label="审核决定",
                                    value="approved"
                                )

                                reviewer_input = gr.Textbox(
                                    label="审核员姓名",
                                    placeholder="请输入您的姓名"
                                )

                            with gr.Column(scale=2):
                                modified_input = gr.Textbox(
                                    label="修改后的答案（仅在修改模式下必填）",
                                    lines=6,
                                    placeholder="如需修改，请在此输入修正后的答案...",
                                    visible=True
                                )

                                comment_input = gr.Textbox(
                                    label="审核意见（可选）",
                                    lines=2,
                                    placeholder="说明审核原因或建议..."
                                )

                        submit_btn = gr.Button("📤 提交审核结果", variant="primary")

            # 操作结果提示
            result_msg = gr.Textbox(
                label="操作结果",
                interactive=False,
                value=""
            )

            # 事件绑定
            def on_select_task(evt: gr.SelectData):
                """点击表格行选择任务"""
                if evt.index[0] >= 0:
                    # evt.value 直接是选中行的数据（列表）
                    selected_row = evt.value
                    if isinstance(selected_row, list) and len(selected_row) > 0:
                        return selected_row[0]  # 第一列是task_id
                return ""

            task_table.select(
                fn=on_select_task,
                outputs=[selected_task]
            )

            selected_task.change(
                fn=self.load_task_detail,
                inputs=[selected_task],
                outputs=[
                    detail_group, query_text, answer_text, reference_docs,
                    conf_score, hall_score, rel_score, risk_alert
                ]
            )

            refresh_btn.click(
                fn=self.load_pending_list,
                outputs=[task_table]
            ).then(
                fn=lambda: gr.update(value=""),  # 清空选择
                outputs=[selected_task]
            )

            submit_btn.click(
                fn=self.submit_review_action,
                inputs=[action_radio, modified_input, comment_input, reviewer_input],
                outputs=[result_msg, stats_text]
            ).then(
                fn=self.load_pending_list,  # 刷新列表
                outputs=[task_table]
            ).then(
                fn=lambda: "",  # 清空选择
                outputs=[selected_task]
            ).then(
                fn=lambda: gr.update(visible=False),  # 隐藏详情
                outputs=[detail_group]
            )

            # 初始化加载
            demo.load(
                fn=self.load_pending_list,
                outputs=[task_table]
            )

        return demo


if __name__ == "__main__":
    # 检查后端连接和审核功能状态
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data.get("human_review_enabled"):
                print("✅ 人机协作功能已启用")
                print(f"   待审核任务数: {data.get('pending_reviews', 0)}")
            else:
                print("⚠️ 人机协作功能未启用")
        else:
            print("⚠️ 健康检查失败")
    except Exception as e:
        print(f"❌ 无法连接到API: {e}")

    # 启动界面
    ui = ReviewerInterface()
    demo = ui.create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # 审核界面用 7861
        share=False,
        show_error=True
    )