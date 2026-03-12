import os

# 设置环境
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# 引入我们之前的所有组件
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import MarkdownHeaderTextSplitter
# 引入新武器
from rank_bm25 import BM25Okapi
from FlagEmbedding import FlagReranker

# 1. 加载所有必要的模型和数据
print("1. 正在加载模型与数据...")
load_dotenv()
api_key = os.getenv("DEEPSEEK_API_KEY")

# 加载 RAG 所需的各种模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
# 加载【精排模型】，第一次运行会自动下载（约2.2GB），请耐心等待
reranker = FlagReranker('BAAI/bge-reranker-large', use_fp16=True)
llm = ChatOpenAI(model="deepseek-chat", api_key=api_key, base_url="https://api.deepseek.com")

# 加载并切分文档（这里我们直接在代码里做，方便演示）
with open("sample_doc.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
all_docs = markdown_splitter.split_text(markdown_document)
doc_contents = [doc.page_content for doc in all_docs]

# 加载向量数据库
vector_db = Chroma(persist_directory="./local_kb", embedding_function=embeddings)

# 准备 BM25 关键词检索器
# BM25 需要先对所有文档进行分词
tokenized_corpus = [list(doc) for doc in doc_contents]  # 最简单的分词：按字分
bm25 = BM25Okapi(tokenized_corpus)


# 2. 定义【混合检索 + 重排】的核心函数
def hybrid_search_and_rerank(query, top_n=5, rerank_top_k=3):
    print(f"\n🔍 收到查询: {query}")

    # --- 第一阶段：粗排 (Recall) ---
    # a. 关键词检索 (BM25)
    tokenized_query = list(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    bm25_top_n_indices = bm25_scores.argsort()[-top_n:][::-1]
    bm25_results = [all_docs[i] for i in bm25_top_n_indices]
    print(f"  - [BM25 召回] 命中 {len(bm25_results)} 条结果。")

    # b. 向量检索 (Vector Search)
    vector_results = vector_db.similarity_search(query, k=top_n)
    print(f"  - [向量召回] 命中 {len(vector_results)} 条结果。")

    # 合并两路结果并去重
    all_recalled_docs = {doc.page_content: doc for doc in bm25_results + vector_results}.values()
    print(f"  - [合并去重后] 总计 {len(all_recalled_docs)} 条候选结果。")

    # --- 第二阶段：精排 (Rerank) ---
    if not all_recalled_docs:
        return ""

    print(f"\n🚀 开始精排 (Reranking)...")
    # 构造 Reranker 需要的句子对
    pairs = [[query, doc.page_content] for doc in all_recalled_docs]

    # 使用精排模型计算相关性得分
    scores = reranker.compute_score(pairs)

    # 将得分和文档绑定，并按分排序
    doc_with_scores = list(zip(all_recalled_docs, scores))
    doc_with_scores.sort(key=lambda x: x[1], reverse=True)

    # 选出最终的 Top-K 结果
    final_results = doc_with_scores[:rerank_top_k]
    print(f"  - [精排完成] 选出 Top {len(final_results)} 条最相关结果。")

    # 拼接最终的上下文
    context_text = "\n\n---\n\n".join([doc.page_content for doc, score in final_results])
    return context_text


# 3. 主流程
user_question = "安装 nginx 时如果遇到依赖冲突怎么办？"

# 调用我们的 Pro 版检索函数
final_context = hybrid_search_and_rerank(user_question)

# --- 后续流程和之前一样 ---
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "你是一个 openEuler 资深运维专家。请根据以下【参考资料】回答用户的问题。"),
    ("user", "【参考资料】\n{context}\n\n【用户问题】\n{question}")
])

chain = prompt_template | llm
response = chain.invoke({"context": final_context, "question": user_question})

print("\n\n✅ 最终回答:\n")
print(response.content)