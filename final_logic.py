import os

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.tools import tool
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter
# 进阶检索组件
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker

# 1. 初始化核心组件
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

print("⚙️ 系统初始化：加载 Embedding 和 Reranker 模型...")
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 只有第一次运行会下载
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)

# 2. 准备知识库 (RAG)
print("📚 系统初始化：构建双路召回索引...")
with open("sample_doc.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
all_docs = markdown_splitter.split_text(markdown_document)
doc_contents = [doc.page_content for doc in all_docs]

# 向量库
vector_db = Chroma(persist_directory="./local_kb", embedding_function=embeddings)
# BM25索引
tokenized_corpus = [list(doc) for doc in doc_contents]
bm25 = BM25Okapi(tokenized_corpus)


# 3. 定义高阶检索函数
def hybrid_search_rerank(query, top_n=5, rerank_top_k=3):
    # a. BM25 召回
    tokenized_query = list(query)
    bm25_indices = bm25.get_scores(tokenized_query).argsort()[-top_n:][::-1]
    bm25_results = [all_docs[i] for i in bm25_indices]

    # b. 向量召回
    vector_results = vector_db.similarity_search(query, k=top_n)

    # 合并去重
    all_recalled_docs = {doc.page_content: doc for doc in bm25_results + vector_results}.values()

    # c. Rerank 精排
    pairs = [[query, doc.page_content] for doc in all_recalled_docs]
    scores = reranker.compute_score(pairs)
    doc_with_scores = list(zip(all_recalled_docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)

    final_results = doc_with_scores[:rerank_top_k]
    return "\n\n---\n\n".join([doc.page_content for doc, score in final_results])


# 4. 定义 Agent 工具
@tool
def search_knowledge_base(query: str):
    """【知识库】查询 openEuler 相关的报错、安装、配置知识。"""
    print(f"🔍 触发高级检索: {query}")
    return hybrid_search_rerank(query)


@tool
def execute_system_command(command: str):
    """【系统执行】查询内存、磁盘等实时系统状态。"""
    print(f"⚡ 执行系统命令: {command}")
    # 保持模拟，以后可以扩展为 Docker
    if "free" in command: return "Mem: 8G Total, 4G Used, 4G Free"
    if "df" in command: return "Disk: 100G Total, 40G Used, 60G Free"
    return f"命令 {command} 已执行。"


tools = [search_knowledge_base, execute_system_command]

# 5. 初始化 Agent 执行器
llm = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个 openEuler 高级运维专家。请自主选择工具解决问题。"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
# 导出这个对象给 main.py 使用
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print("✅ 系统逻辑加载完成，等待接口调用...")