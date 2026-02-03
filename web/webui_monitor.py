# webui_monitor.py - 生产监控仪表盘
import os
import requests
import gradio as gr
from datetime import datetime

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")


class MonitorDashboard:
    """生产监控仪表盘"""

    def __init__(self):
        self.api_url = API_URL

    def fetch_dashboard(self):
        """获取仪表盘数据"""
        try:
            r = requests.get(f"{self.api_url}/api/monitor/dashboard", timeout=5)
            if r.status_code == 200:
                return r.json()
            return {"error": "API错误"}
        except Exception as e:
            return {"error": str(e)}

    def fetch_alerts(self):
        """获取告警"""
        try:
            r = requests.get(f"{self.api_url}/api/monitor/alerts", timeout=5)
            if r.status_code == 200:
                return r.json()
            return {"alerts": []}
        except:
            return {"alerts": []}

    def create_interface(self):
        """创建监控界面"""
        with gr.Blocks(title="Self-RAG 生产监控", theme=gr.themes.Soft()) as demo:
            gr.Markdown("""
            # 📊 Self-RAG 生产监控仪表盘
            实时监控大模型服务性能与质量指标
            """)

            # 自动刷新组件
            with gr.Row():
                refresh_btn = gr.Button("🔄 刷新数据", variant="primary")
                auto_refresh = gr.Checkbox(label="自动刷新(5s)", value=True)

            # 概览指标
            with gr.Row():
                total_queries = gr.Number(label="总查询数", value=0, interactive=False)
                avg_confidence = gr.Number(label="平均置信度", value=0, interactive=False)
                avg_latency = gr.Number(label="平均响应(ms)", value=0, interactive=False)
                error_rate = gr.Number(label="错误率", value=0, interactive=False)

            # 模型分布
            with gr.Row():
                model_dist = gr.JSON(label="模型使用分布", value={})

            # 实时告警
            with gr.Row():
                alerts_table = gr.Dataframe(
                    headers=["时间", "类型", "查询", "严重程度"],
                    label="⚠️ 实时告警",
                    row_count=5
                )

            # 最近查询
            with gr.Row():
                history_table = gr.Dataframe(
                    headers=["时间", "模型", "置信度", "耗时(ms)", "状态"],
                    label="最近查询",
                    row_count=10
                )

            # 更新函数
            def update_data():
                dashboard = self.fetch_dashboard()
                alerts = self.fetch_alerts()

                if "error" in dashboard:
                    return [0, 0, 0, 0, {}, [], []]

                overview = dashboard.get("overview", {})

                # 格式化告警
                alerts_data = []
                for a in alerts.get("alerts", [])[:5]:
                    alerts_data.append([
                        datetime.fromtimestamp(a["timestamp"]).strftime("%H:%M:%S"),
                        a["type"],
                        a["query"],
                        a["severity"]
                    ])

                # 格式化历史
                history_data = []
                for h in dashboard.get("recent_history", []):
                    history_data.append([
                        datetime.fromtimestamp(h["timestamp"]).strftime("%H:%M:%S"),
                        h["model"],
                        f"{h['confidence']:.2f}",
                        f"{h['total_duration_ms']:.0f}",
                        h["status"]
                    ])

                return [
                    overview.get("total_queries", 0),
                    overview.get("avg_confidence", 0),
                    overview.get("avg_response_time_ms", 0),
                    overview.get("error_rate", 0),
                    dashboard.get("model_distribution", {}),
                    alerts_data,
                    history_data
                ]

            # 事件绑定
            refresh_btn.click(
                fn=update_data,
                outputs=[total_queries, avg_confidence, avg_latency, error_rate,
                         model_dist, alerts_table, history_table]
            )

            # 自动刷新
            demo.load(
                fn=update_data,
                outputs=[total_queries, avg_confidence, avg_latency, error_rate,
                         model_dist, alerts_table, history_table],
                every=5  # 每5秒刷新
            )

        return demo


if __name__ == "__main__":
    dashboard = MonitorDashboard()
    demo = dashboard.create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7862,  # 监控用7862端口
        share=False
    )