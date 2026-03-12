import os

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
# 复用之前的 RAG 组件
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. 加载配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

print("正在初始化 AI 运维专家（加载大模型 + 知识库）...")

# 2. 准备 RAG 知识库（作为 Agent 的大脑外挂）
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./local_kb", embedding_function=embeddings)


# 3. 定义工具集 (Toolbox)

@tool
def search_knowledge_base(query: str):
    """
    【知识库工具】
    当用户询问关于 openEuler 的配置方法、报错解决方案、安装指南等知识类问题时，
    必须优先调用此工具。
    输入：用户的自然语言问题。
    返回：相关的文档片段。
    """
    print(f"\n📚 [RAG] 正在检索知识库: {query}")
    docs = vector_db.similarity_search(query, k=2)
    return "\n\n".join([d.page_content for d in docs])


@tool
def execute_system_command(command: str):
    """
    【系统执行工具】
    当用户要求查询系统状态（如内存、磁盘、进程、网络）时调用。
    注意：此工具只能执行查询类命令（free, df, top, ip addr 等）。
    输入：Linux Shell 命令。
    """
    print(f"\n⚡ [Shell] 正在执行命令: {command}")
    # 这里我们做简单的模拟，你可以换成 subprocess.run
    if "free" in command:
        return "              total        used        free\nMem:        8123456     4123456     4000000"
    elif "df" in command:
        return "Filesystem     Size   Used  Avail Use% Mounted on\n/dev/sda1      100G    40G    60G  40% /"
    else:
        return f"命令 {command} 执行成功（模拟数据）。"


# 工具打包
tools = [search_knowledge_base, execute_system_command]

# 4. 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    temperature=0  # 设为0，让它更严谨，不要瞎编
)

# 5. 创建 Agent
prompt = ChatPromptTemplate.from_messages([
    ("system", """你是一个 openEuler 高级运维专家 Agent。
    你拥有【查阅内部知识库】和【执行系统命令】的能力。

    请遵循以下思考逻辑：
    1. 如果用户问的是报错、配置、安装方法 -> 请调用 search_knowledge_base。
    2. 如果用户问的是当前系统状态（内存、磁盘） -> 请调用 execute_system_command。
    3. 如果都不需要，直接陪用户聊天。
    """),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 6. 开始测试！
print("\n========== 测试场景 1：问知识 ==========")
agent_executor.invoke({"input": "我安装 nginx 报错依赖冲突，该怎么解决？"})

print("\n========== 测试场景 2：查系统 ==========")
agent_executor.invoke({"input": "帮我看看现在内存还剩多少？"})