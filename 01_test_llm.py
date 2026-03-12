import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 文件里的 API Key
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 初始化大模型 (用 DeepSeek 的接口)
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com",
    max_tokens=1024
)

print("正在呼叫大模型...")
# 问它一个关于 Linux 的问题
response = llm.invoke("请用一句话解释一下什么是 openEuler？")

print("\n大模型的回答：")
print(response.content)