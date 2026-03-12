# openEuler AI DevOps Assistant

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95%2B-green.svg)](https://fastapi.tiangolo.com/)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-purple.svg)](https://www.langchain.com/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

基于 LLM Agent 与混合检索 RAG 架构的 openEuler 智能运维与开发辅助系统。

本项目旨在通过自然语言交互降低 openEuler 系统的运维门槛。系统底层结合了 DeepSeek 大模型的推理规划能力与本地化知识库，能够自动将自然语言意图转化为精确的 Linux 运维指令并安全执行，同时提供高准确率的 openEuler 社区文档问答支持。

## 核心特性 / Features

- **混合检索 RAG 架构**
  弃用单一向量检索，采用 `BM25 稀疏检索` + `BGE Dense 向量检索` 的双路召回策略。配合 `BGE-Reranker-large` 模型进行二次重排（Cross-Encoder），有效解决了 openEuler 专有名词、参数缩写在传统 RAG 中召回率低的问题。
  
- **Tool-Calling Agent 意图路由**
  基于 LangChain 实现了动态工具调用机制。Agent 根据用户 Query 上下文，自主决策路由至“知识库问答检索（RAG Tool）”或“系统状态探测（CMD Tool）”，实现多轮对话下的复杂运维意图拆解。
  
- **OpenAI 兼容 API 层**
  后端通过 FastAPI 实现异步服务，并在中间件层完成了对 OpenAI `/v1/chat/completions` 协议的完全适配与伪装。可无缝对接 ChatGPT-Next-Web、LobeChat 等任意主流开源前端生态。
  
- **开箱即用的容器化部署**
  前后端完全解耦，提供极简的 Docker 启动方案，支持跨平台快速部署。

## 系统架构 / Architecture

![架构图](https://github.com/yoyochilli/openeuler_ai_devops/architecture.png)

## 技术栈 / Tech Stack

- **LLM**: DeepSeek
- **AI 框架**: LangChain
- **Web 框架**: FastAPI
- **向量检索**: FAISS
- **稀疏检索**: Rank-BM25
- **Embedding 模型**: BAAI/bge-small-zh-v1.5
- **Reranker 模型**: BAAI/bge-reranker-large

## 快速开始 / Quick Start

### 环境依赖 / Prerequisites
- Python >= 3.10
- Docker & Docker Compose (推荐用于前端部署)

### 1. 后端部署 / Backend Setup

**a. 克隆项目并安装依赖**
```bash
# 克隆项目
git clone https://github.com/YourUsername/openeuler_ai_devops.git
cd openeuler_ai_devops

# 创建并激活虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

**b. 配置环境变量**
```bash
# 从模板复制环境变量文件
cp .env.example .env
```
然后编辑 `.env` 文件，填入你的配置信息，至少需要填写以下两项：
```dotenv
DEEPSEEK_API_KEY="sk-xxxxxxxxxxxxxxxxxxxx"
EMBEDDING_MODEL_PATH="BAAI/bge-small-zh-v1.5"
```

**c. 启动后端服务**
```bash
# 启动 FastAPI 服务，默认监听 8000 端口
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```
服务启动后，你可以在 `http://localhost:8000/docs` 查看 API 文档。

### 2. 前端部署 / Frontend Setup
推荐使用 [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web) 作为交互终端。通过 Docker 一键拉取并配置反向代理指向本地 API 服务：
```bash
docker run -d -p 3000:3000 \
  -e OPENAI_API_KEY="sk-any-string" \
  -e BASE_URL="http://<你的局域网IP>:8000" \
  -e CUSTOM_MODELS="-all,+gpt-3.5-turbo@openEuler-Agent" \
  yidadaa/chatgpt-next-web
```
**重要提示**:
- 请将 `<你的局域网IP>` 替换为你运行后端服务电脑的局域网 IP 地址（例如 `192.168.1.10`）。
- **请勿** 使用 `localhost` 或 `127.0.0.1`，因为 Docker 容器有自己独立的网络空间，使用 `localhost` 会指向容器自身，导致无法访问到宿主机（你的电脑）上运行的后端服务。
- 你可以通过在终端运行 `ifconfig` (macOS/Linux) 或 `ipconfig` (Windows) 来查找你的局域网 IP。

部署完成后，在浏览器中打开 `http://localhost:3000` 即可开始与你的 openEuler AI 助手对话。

## 许可证 / License
本项目采用 [Apache 2.0 license](./LICENSE) 开源许可证。