from langchain.text_splitter import MarkdownHeaderTextSplitter

# 1. 读取我们的模拟文档
with open("sample_doc.md", "r", encoding="utf-8") as f:
    markdown_document = f.read()

# 2. 定义切分规则（保留一、二、三级标题作为标签/Metadata）
headers_to_split_on = [
    ("#", "Header 1"),
    ("##", "Header 2"),
    ("###", "Header 3"),
]

# 3. 初始化切分器并切分文档
markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
md_header_splits = markdown_splitter.split_text(markdown_document)

# 4. 打印结果，见证奇迹
print(f"一共被切分成了 {len(md_header_splits)} 个块。\n")

for i, chunk in enumerate(md_header_splits):
    print(f"--- 第 {i+1} 块 ---")
    print(f"内容: {chunk.page_content}")
    print(f"标签(Metadata): {chunk.metadata}\n")