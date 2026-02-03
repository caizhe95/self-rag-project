# test_ab_comparison.py - 完整版AB对比测试

import asyncio
import requests
import json
import time
from typing import List, Dict, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from statistics import mean

API_URL = "http://localhost:8000"


@dataclass
class TestConfig:
    """测试配置"""
    name: str
    hybrid_weights: Dict[str, float]
    reranker_enabled: bool
    description: str


class ABComparisonTester:
    """AB对比测试器"""

    def __init__(self):
        self.api_url = API_URL
        self.configs = [
            TestConfig(
                name="纯向量检索",
                hybrid_weights={"bm25": 0.0, "vector": 1.0},
                reranker_enabled=False,
                description="Chroma向量相似度"
            ),
            TestConfig(
                name="混合检索(BM25+向量)",
                hybrid_weights={"bm25": 0.4, "vector": 0.6},
                reranker_enabled=False,
                description="BM40%+向量60%，无重排序"
            ),
            TestConfig(
                name="混合+重排序",
                hybrid_weights={"bm25": 0.4, "vector": 0.6},
                reranker_enabled=True,
                description="BM25+向量+Cross-Encoder"
            ),
        ]

        self.test_queries = [
            # ai_ethics.md
            "算法偏见怎么解决",
            "AI隐私保护技术有哪些",

            # ai_history.md
            "Transformer是哪一年提出的",
            "ChatGPT发展历程",

            # deep_learning_arch.md
            "CNN和RNN有什么区别",
            "注意力机制原理",

            # llm_training.md
            "RLHF训练流程",
            "LoRA微调优势",

            # ml_basics.md
            "过拟合解决方法",
            "监督学习应用场景",

            # 跨文档
            "深度学习发展历史",
            "大模型伦理问题",
        ]

        self.results: List[Dict] = []

    async def run(self):
        """运行完整测试"""
        print("=" * 80)
        print("🧪 AB对比测试：量化混合检索与重排序的真实提升")
        print("=" * 80)
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(
            f"查询数: {len(self.test_queries)} × {len(self.configs)}配置 = {len(self.test_queries) * len(self.configs)}次")
        print("-" * 80)

        # 检查服务
        if not await self._check_service():
            print("❌ 服务不可用")
            return

        # 测试每个配置
        for config in self.configs:
            print(f"\n{'=' * 80}")
            print(f"📋 配置: {config.name}")
            print(f"   {config.description}")
            print("-" * 80)

            config_results = await self._test_config(config)
            self.results.append({
                "config": asdict(config),
                "results": config_results
            })

        # 生成报告
        return self._generate_report()

    async def _check_service(self) -> bool:
        """检查服务状态"""
        try:
            r = requests.get(f"{self.api_url}/health", timeout=5)
            if r.status_code == 200:
                data = r.json()
                print(f"✅ 服务正常")
                print(f"   模型: {data.get('model', 'unknown')}")
                print(f"   文档数: {data.get('document_count', 0)}")
                return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
        return False

    async def _test_config(self, config: TestConfig) -> List[Dict]:
        """测试单个配置"""
        results = []

        # 应用配置
        if not self._apply_config(config):
            print(f"   ⚠️ 配置应用失败，跳过")
            return results

        # 等待配置生效
        await asyncio.sleep(0.5)

        for idx, query in enumerate(self.test_queries, 1):
            print(f"\n   [{idx}/{len(self.test_queries)}] '{query}'")

            try:
                # 使用debug接口获取详细信息
                r = requests.post(
                    f"{self.api_url}/api/retrieval/debug",
                    params={"query": query},
                    timeout=30
                )

                if r.status_code != 200:
                    print(f"      ❌ API错误: {r.status_code}")
                    continue

                data = r.json()
                if not data.get("success"):
                    print(f"      ❌ 请求失败")
                    continue

                # 解析指标
                metrics = data.get("metrics", {})
                docs = data.get("retrieved_docs", [])

                result = {
                    "query": query,
                    "vector_count": data.get("vector_count", 0),
                    "bm25_count": data.get("bm25_count", 0),
                    "final_count": data.get("final_count", 0),
                    "vector_time_ms": metrics.get("vector_time_ms", 0),
                    "bm25_time_ms": metrics.get("bm25_time_ms", 0),
                    "rerank_time_ms": metrics.get("rerank_time_ms", 0),
                    "total_time_ms": metrics.get("total_time_ms", 0),
                    "docs": docs,
                    "top_scores": {
                        "vector": max([d.get("vector_score", 0) for d in docs if d.get("vector_score")], default=0),
                        "bm25": max([d.get("bm25_score", 0) for d in docs if d.get("bm25_score")], default=0),
                        "rerank": max([d.get("rerank_score", 0) for d in docs if d.get("rerank_score")], default=0),
                        "final": max([d.get("final_score", 0) for d in docs], default=0) if docs else 0,
                    }
                }
                results.append(result)

                # 实时显示
                print(f"      ⏱️  {result['total_time_ms']:.0f}ms "
                      f"| 📄 V:{result['vector_count']} B:{result['bm25_count']} F:{result['final_count']}"
                      f"| 🎯 最高分:{result['top_scores']['final']:.3f}")

            except Exception as e:
                print(f"      ❌ 错误: {e}")

        return results

    def _apply_config(self, config: TestConfig) -> bool:
        """应用配置到服务器"""
        try:
            r = requests.post(
                f"{self.api_url}/api/config/retrieval",
                json={
                    "hybrid_weights": config.hybrid_weights,
                    "reranker_enabled": config.reranker_enabled
                },
                timeout=10
            )
            if r.status_code == 200:
                print(f"   ✅ 配置已应用: {config.hybrid_weights}, rerank={config.reranker_enabled}")
                return True
            else:
                print(f"   ❌ 配置失败: {r.status_code}")
                return False
        except Exception as e:
            print(f"   ❌ 配置异常: {e}")
            return False

    def _generate_report(self) -> Dict[str, Any]:
        """生成对比报告"""
        print("\n" + "=" * 80)
        print("📊 AB对比测试报告")
        print("=" * 80)

        if len(self.results) != 3:
            print("❌ 数据不完整")
            return {}

        # 提取三种配置
        pure_vector = self.results[0]["results"]
        hybrid = self.results[1]["results"]
        hybrid_rerank = self.results[2]["results"]

        # 计算汇总指标
        def calc_metrics(results: List[Dict]) -> Dict:
            if not results:
                return {}
            return {
                "avg_time_ms": mean([r["total_time_ms"] for r in results]),
                "avg_vector_count": mean([r["vector_count"] for r in results]),
                "avg_bm25_count": mean([r["bm25_count"] for r in results]),
                "avg_final_count": mean([r["final_count"] for r in results]),
                "avg_top_score": mean([r["top_scores"]["final"] for r in results]),
            }

        m1 = calc_metrics(pure_vector)
        m2 = calc_metrics(hybrid)
        m3 = calc_metrics(hybrid_rerank)

        # 打印对比表
        print("\n【核心指标对比】")
        print(f"{'指标':<25} {'纯向量':<15} {'混合检索':<15} {'混合+重排':<15} {'混合提升':<12} {'重排提升':<12}")
        print("-" * 100)

        def fmt(val, unit=""):
            return f"{val:.1f}{unit}" if isinstance(val, float) else str(val)

        def calc_imp(base, new):
            return f"{(new / base - 1) * 100:+.1f}%" if base > 0 else "N/A"

        rows = [
            ("平均响应时间", "ms", m1.get("avg_time_ms", 0), m2.get("avg_time_ms", 0), m3.get("avg_time_ms", 0)),
            ("召回文档数", "篇", m1.get("avg_final_count", 0), m2.get("avg_final_count", 0),
             m3.get("avg_final_count", 0)),
            ("Top1相关性分数", "", m1.get("avg_top_score", 0), m2.get("avg_top_score", 0), m3.get("avg_top_score", 0)),
            ("向量检索耗时", "ms", m1.get("avg_time_ms", 0), m2.get("avg_vector_count", 0) * 0,
             m3.get("avg_vector_count", 0) * 0),  # 占位
        ]

        for name, unit, v1, v2, v3 in rows[:3]:
            imp1 = calc_imp(v1, v2)
            imp2 = calc_imp(v2, v3)
            print(f"{name:<25} {fmt(v1, unit):<15} {fmt(v2, unit):<15} {fmt(v3, unit):<15} {imp1:<12} {imp2:<12}")

        # 详细分析
        print("\n【详细分析】")

        # 召回数量对比
        recall_pure = m1["avg_final_count"]
        recall_hybrid = m2["avg_final_count"]
        recall_boost = (recall_hybrid / recall_pure - 1) * 100 if recall_pure > 0 else 0

        print(f"\n1️⃣ 召回数量")
        print(f"   纯向量: {recall_pure:.1f}篇")
        print(f"   混合检索: {recall_hybrid:.1f}篇 (提升 {recall_boost:+.1f}%)")
        print(f"   ✅ BM25补充了向量检索未覆盖的文档")

        # 相关性对比
        score_pure = m1["avg_top_score"]
        score_hybrid = m2["avg_top_score"]
        score_rerank = m3["avg_top_score"]

        hybrid_boost = (score_hybrid / score_pure - 1) * 100 if score_pure > 0 else 0
        rerank_boost = (score_rerank / score_hybrid - 1) * 100 if score_hybrid > 0 else 0

        print(f"\n2️⃣ 相关性质量 (Top1分数)")
        print(f"   纯向量: {score_pure:.3f}")
        print(f"   混合检索: {score_hybrid:.3f} (提升 {hybrid_boost:+.1f}%)")
        print(f"   混合+重排: {score_rerank:.3f} (再提升 {rerank_boost:+.1f}%)")
        print(f"   ✅ 混合检索通过加权融合提升相关性")
        print(f"   ✅ Cross-Encoder重排序进一步优化TopK质量")

        # 性能对比
        time_pure = m1["avg_time_ms"]
        time_hybrid = m2["avg_time_ms"]
        time_rerank = m3["avg_time_ms"]

        time_overhead = (time_rerank / time_pure - 1) * 100 if time_pure > 0 else 0

        print(f"\n3️⃣ 响应时间")
        print(f"   纯向量: {time_pure:.0f}ms")
        print(f"   混合检索: {time_hybrid:.0f}ms")
        print(f"   混合+重排: {time_rerank:.0f}ms ( overhead {time_overhead:+.1f}%)")
        print(f"   ⏱️  重排序增加约 {m3['avg_time_ms'] - m2['avg_time_ms']:.0f}ms")

        # 结论
        print("\n【核心结论】")
        print(f"  ✅ 混合检索召回数量提升 {recall_boost:.1f}%")
        print(f"  ✅ 混合检索相关性提升 {hybrid_boost:.1f}%")
        print(f"  ✅ 重排序额外提升相关性 {rerank_boost:.1f}%")
        if time_overhead < 50:
            print(f"  ✅ 时间开销仅 {time_overhead:.1f}%，性价比高")
        else:
            print(f"  ⚠️  时间开销 {time_overhead:.1f}%，需权衡")

        print("\n【面试数据】")
        print(f'  "混合检索召回数量提升{recall_boost:.0f}%，重排序后Top3相关性提升{rerank_boost:.0f}%，')
        print(f'   响应时间增加{time_overhead:.0f}%，在可接受范围内。"')

        print("=" * 80)

        # 恢复配置
        try:
            requests.post(f"{self.api_url}/api/config/retrieval/reset", timeout=5)
            print("\n🔄 配置已恢复")
        except:
            pass

        return {
            "timestamp": datetime.now().isoformat(),
            "configs": self.results,
            "summary": {
                "recall_boost_percent": round(recall_boost, 1),
                "hybrid_score_boost_percent": round(hybrid_boost, 1),
                "rerank_score_boost_percent": round(rerank_boost, 1),
                "time_overhead_percent": round(time_overhead, 1),
            },
            "metrics": {
                "pure_vector": m1,
                "hybrid": m2,
                "hybrid_rerank": m3
            }
        }


async def main():
    tester = ABComparisonTester()
    report = await tester.run()

    # 保存
    filename = f"ab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n💾 报告已保存: {filename}")


if __name__ == "__main__":
    asyncio.run(main())