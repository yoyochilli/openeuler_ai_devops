import os
# 【防掉线神器】设置国内 HuggingFace 镜像源，防止网络超时！
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

print("1. 正在读取并切分文档...")
# 复用我们上一步的切分代码
with open("sample_doc.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

headers_to_split_on = [("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
docs = markdown_splitter.split_text(markdown_document)

print("2. 正在加载 BGE 向量化模型 (第一次运行会自动下载，约 100MB，请耐心等待)...")
# 加载 HuggingFace 上的开源 Embedding 模型
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")

print("3. 正在将文档转换为向量，并存入本地 Chroma 数据库...")
# 创建 Chroma 数据库，存放在当前目录下的 "local_kb" 文件夹中
vector_db = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./local_kb"  # 持久化保存到本地
)

print("✅ 知识库构建完成！你会在左侧看到多了一个 'local_kb' 文件夹。")