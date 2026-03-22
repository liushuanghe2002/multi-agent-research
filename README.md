# Multi-Agent Research Assistant

一个基于 LangGraph 的多智能体研究助手，包含 Researcher、Analyzer、Writer 三个协作 Agent。

## 功能特性

- 🔍 **Researcher Agent**: 使用 Tavily 搜索相关资料
- 📊 **Analyzer Agent**: 分析搜索结果，提取关键信息
- ✍️ **Writer Agent**: 生成结构化研究报告

## 技术栈

- LangGraph - 多智能体编排框架
- LangChain Groq - LLM 推理（Llama3-70B）
- Tavily - 搜索引擎 API

## 快速开始

### 1. 克隆项目

```bash
git clone <your-repo-url>
cd multi-agent-research
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境变量

创建 `.env` 文件：

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

获取 API Key：
- **Groq**: https://console.groq.com/keys
- **Tavily**: https://tavily.com (免费版每月1000次)

### 4. 运行

```bash
python research_agent.py
```

## 在线学习

本项目包含详细的代码解析页面，访问：
- `code-guide.html` - 交互式代码学习指南

## 项目结构

```
multi-agent-research/
├── research_agent.py   # 核心代码
├── requirements.txt    # 依赖
├── .env.example        # 环境变量模板
├── README.md          # 本文件
└── code-guide.html    # 代码解析页面
```

## 许可证

MIT License
