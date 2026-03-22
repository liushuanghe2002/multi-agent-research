"""
Multi-Agent Research Assistant
===============================
A LangGraph-powered research assistant with three collaborating AI agents:
  - Researcher:  Searches the web for information using Tavily
  - Analyst:     Extracts key insights from the research
  - Summariser:  Produces a clean, structured summary

Usage:
    1. Copy .env.example to .env and fill in your API keys
    2. pip install -r requirements.txt
    3. python research_agent.py
"""

import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END
from groq import Groq
from tavily import TavilyClient
from dotenv import load_dotenv

# ─── Configuration ────────────────────────────────────────────────────────────
load_dotenv()

GROQ_API_KEY   = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
MODEL          = "llama-3.1-8b-instant"   # 免费，延迟极低 (~200ms)


# ─── Shared State Schema ──────────────────────────────────────────────────────
# TypedDict 定义整个工作流的"共享黑板"：
#   每个 Agent 只读取自己需要的字段，只写入自己负责的字段。
class State(TypedDict):
    topic:    str   # 输入：用户研究主题
    research: str   # Researcher  写入
    analysis: str   # Analyst     写入
    summary:  str   # Summariser  写入


# ─── Agent 1: Researcher ──────────────────────────────────────────────────────
def research_node(state: State) -> dict:
    """
    职责：调用 Tavily 搜索 API，获取原始网络文章。
    输入：state["topic"]
    输出：{"research": <合并后的原始文章文本>}
    """
    print("📚 Researching...")
    tavily = TavilyClient(api_key=TAVILY_API_KEY)
    results = tavily.search(state["topic"], max_results=5)

    # 将 5 篇文章的标题+内容拼接成一段大文本，传给下一个 Agent
    combined_results = ""
    for r in results["results"]:
        combined_results += r["title"] + "\n" + r["content"] + "\n\n"

    print(f"  Found {len(results['results'])} articles")
    return {"research": combined_results}


# ─── Agent 2: Analyst ─────────────────────────────────────────────────────────
def analyst_node(state: State) -> dict:
    """
    职责：调用 Groq LLM，对原始搜索结果进行分析，提炼关键见解。
    输入：state["topic"] + state["research"]
    输出：{"analysis": <结构化分析文本>}
    """
    print("📊 Analysing...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a research analyst. You are given raw search results "
                    "on a topic. Extract the key insights, identify the most important "
                    "facts, and note any conflicting information between sources. "
                    "Be thorough and objective."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\nResearch:\n{state['research']}",
            },
        ],
    )

    answer = response.choices[0].message.content
    print("  ✓ Analysis complete")
    return {"analysis": answer}


# ─── Agent 3: Summariser ──────────────────────────────────────────────────────
def summariser_node(state: State) -> dict:
    """
    职责：调用 Groq LLM，将分析结果转化为面向非专家的可读报告。
    输入：state["topic"] + state["analysis"]
    输出：{"summary": <最终总结报告>}
    """
    print("💬 Writing summary...")
    groq_client = Groq(api_key=GROQ_API_KEY)

    response = groq_client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a technical writer. You are given an analysis of research "
                    "on a topic. Write a clear, well-structured summary that a "
                    "non-expert could understand. Include key takeaways at the end."
                ),
            },
            {
                "role": "user",
                "content": f"Topic: {state['topic']}\n\nAnalysis:\n{state['analysis']}",
            },
        ],
    )

    answer = response.choices[0].message.content
    print("  ✓ Summary complete")
    return {"summary": answer}


# ─── Build the Graph ──────────────────────────────────────────────────────────
def build_graph():
    """
    用 LangGraph 的 StateGraph API 把三个 Agent 连成有向无环图：

        START → research_node → analyst_node → summariser_node → END

    graph.compile() 返回一个可调用的 Runnable，
    支持 .invoke() / .stream() / .astream() 等接口。
    """
    graph = StateGraph(State)

    # 注册节点（名称 → 函数）
    graph.add_node("research",   research_node)
    graph.add_node("analyst",    analyst_node)
    graph.add_node("summariser", summariser_node)

    # 添加有向边（执行顺序）
    graph.add_edge(START,        "research")
    graph.add_edge("research",   "analyst")
    graph.add_edge("analyst",    "summariser")
    graph.add_edge("summariser", END)

    return graph.compile()


# ─── High-level API ───────────────────────────────────────────────────────────
def research(topic: str) -> dict:
    """
    对外暴露的入口函数。
    运行完整流水线，打印各步进度，最终返回完整的 state dict。
    """
    print("=" * 60)
    print(f"RESEARCH AGENT — {topic}")
    print("=" * 60 + "\n")

    app    = build_graph()
    result = app.invoke({"topic": topic})

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(result["summary"])

    return result


# ─── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    topic = input("What would you like to research? ").strip()
    if topic:
        research(topic)
    else:
        print("No topic provided.")
