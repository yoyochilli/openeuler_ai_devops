import os
# 设置国内镜像，防止连不上 HuggingFace
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
# 🌟【关键修改】这里必须改用 langchain_community，避开那个报错的包！
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载配置
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

print("正在唤醒知识库和大脑...")
# 🌟【关键修改】这里也要改，必须和 03 文件保持一致
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
vector_db = Chroma(persist_directory="./local_kb", embedding_function=embeddings)

# 唤醒 DeepSeek 大模型
llm = ChatOpenAI(
    model="deepseek-chat",
    api_key=api_key,
    base_url="https://api.deepseek.com"
)

# 2. 模拟用户提问
user_question = "在 openEuler 系统里，安装 nginx 遇到依赖冲突报错怎么办？"
print(f"\n👨‍💻 用户提问: {user_question}")

# 3. 检索
print("🔍 正在知识库中检索相关方案...")
retrieved_docs = vector_db.similarity_search(user_question, k=2)

# 拼凑上下文
context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
print(f"📄 检索到的参考资料：\n{context_text}\n")

# 4. 生成回答
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 openEuler 资深运维专家。请根据以下【参考资料】回答用户的问题。如果资料里没有提到，你就回答不知道。"),
    ("user", "【参考资料】\n{context}\n\n【用户问题】\n{question}")
])

print("🧠 大模型正在分析并生成回答...")
chain = prompt_template | llm
response = chain.invoke({
    "context": context_text,
    "question": user_question
})

print(f"\n🤖 运维助手最终回答:\n{response.content}")