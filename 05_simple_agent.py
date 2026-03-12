import os
from dotenv import load_dotenv
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# 1. 加载配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")


# 2. 定义工具 (Tools) —— 给 AI 的“双手”
# 🌟 关键点：@tool 装饰器和函数下面的 """注释""" 非常重要！
# 大模型是靠读这些注释，来决定要不要用这个工具，以及怎么用。

@tool
def get_system_version_tool():
    """
    当用户询问 openEuler 系统版本信息时，调用此工具。
    它会模拟执行 'cat /etc/os-release' 命令。
    """
    # 这里我们模拟返回一个结果，实际项目中会替换成真实的 subprocess.run
    return """
    NAME="openEuler"
    VERSION="22.03 (LTS-SP1)"
    ID="openEuler"
    VERSION_ID="22.03"
    PRETTY_NAME="openEuler 22.03 (LTS-SP1)"
    ANSI_COLOR="0;31"
    """


@tool
def execute_linux_command_tool(command: str):
    """
    这是一个万能的 Linux 命令执行工具。
    当用户要求执行具体的命令（如 ls, dnf, systemctl）时调用。
    输入参数 command: 需要执行的 Linux 命令字符串。
    """
    print(f"\n⚡ [系统监控] 正在宿主机上执行命令: {command}")

    # 模拟一些常见命令的返回
    if "date" in command:
        return "2025年 1月 23日 星期四 10:00:00 CST"
    elif "whoami" in command:
        return "root"
    else:
        return f"命令 '{command}' 已执行成功（模拟返回）。"


# 把工具打包成一个列表
tools = [get_system_version_tool, execute_linux_command_tool]

# 3. 初始化大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 4. 创建 Agent (智能体)
# 定义 Prompt：告诉 AI 它是一个拥有工具的运维助手
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个高级 Linux 运维 Agent。你拥有操作系统的权限。请根据用户需求，自主选择最合适的工具去解决问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),  # 这是给 Agent 留的“草稿本”，用来记录它的思考过程
])

# 组装 Agent
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)  # verbose=True 会打印出 AI 的思考过程

# 5. 测试 Agent
print("------ 测试 1：查版本 ------")
agent_executor.invoke({"input": "请帮我查一下当前 openEuler 的系统版本是多少？"})

print("\n------ 测试 2：执行命令 ------")
agent_executor.invoke({"input": "帮我看看现在的系统时间"})
