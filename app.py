import streamlit as st
import os

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 页面基础配置
st.set_page_config(page_title="openEuler 智能运维助手", page_icon="🐧")
st.title("🐧 openEuler AI DevOps 助手")
st.caption("🚀 基于 RAG + Agent 架构 | 支持：文档检索、故障排查、系统监控")


# 2. 核心逻辑封装（加缓存，只加载一次！）
@st.cache_resource
def initialize_agent():
    load_dotenv()
    api_key = os.getenv("DEEPSEEK_API_KEY")

    # --- 加载知识库 ---
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
    vector_db = Chroma(persist_directory="./local_kb", embedding_function=embeddings)

    # --- 定义工具 ---
    @tool
    def search_knowledge_base(query: str):
        """【知识库】查询 openEuler 配置、报错、文档。"""
        docs = vector_db.similarity_search(query, k=2)
        return "\n\n".join([d.page_content for d in docs])

    @tool
    def execute_system_command(command: str):
        """【系统执行】查询内存、磁盘等系统状态。"""
        # 模拟数据
        if "free" in command:
            return "              total        used        free\nMem:        8123456     4123456     4000000"
        return f"命令 {command} 执行成功。"

    tools = [search_knowledge_base, execute_system_command]

    # --- 初始化 Agent ---
    llm = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com", temperature=0)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个 openEuler 高级运维专家 Agent。请根据用户需求，自主选择【查阅知识库】或【执行系统命令】。"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# 初始化（这一步可能会花几秒钟）
if "agent_executor" not in st.session_state:
    with st.spinner("正在启动 AI 引擎..."):
        st.session_state.agent_executor = initialize_agent()

# 3. 聊天界面逻辑
# 初始化历史记录
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入运维问题（例如：安装 nginx 报错 / 查看内存）"):
    # 显示用户的话
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # AI 思考并回答
    with st.chat_message("assistant"):
        with st.spinner("🤖 AI 正在思考与操作..."):
            try:
                # 调用 Agent
                response = st.session_state.agent_executor.invoke({"input": prompt})
                result = response["output"]
                st.markdown(result)
                # 记录 AI 的话
                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"发生错误: {e}")