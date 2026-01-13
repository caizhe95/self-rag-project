import asyncio
from core.rag_chain import SelfRAGChain
from config import Config


async def main():
    config = Config()
    rag = SelfRAGChain(config)

    SAMPLE_DOCS = [
        {
            "text": """LangChain是一个用于开发LLM应用的框架，提供模块化工具和链式调用能力。支持记忆管理、工具集成等功能。""",
            "metadata": {"source": "langchain_intro.txt"}
        },
        {
            "text": """Self-RAG是RAG的增强版本，核心特点是自我评估和动态优化。工作流程包含检索、生成、评估、决策四个阶段，支持循环优化和人工审核。""",
            "metadata": {"source": "self_rag_intro.txt"}
        }
    ]

    # 异步索引
    print("异步索引文档...")
    await rag.aindex_documents(
        texts=[doc["text"] for doc in SAMPLE_DOCS],
        metadatas=[doc["metadata"] for doc in SAMPLE_DOCS]
    )

    # 异步流式查询
    print("\n异步流式查询...")
    async for event in rag.astream_query("什么是Self-RAG？"):
        if event["event"] == "on_chain_stream" and event["name"] == "generate":
            print(event["data"]["chunk"], end="", flush=True)

    print("\n查询完成")


if __name__ == "__main__":
    asyncio.run(main())