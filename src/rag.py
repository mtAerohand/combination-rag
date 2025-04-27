# https://developer.nvidia.com/ja-jp/blog/rag-with-sota-reranking-model-in-japanese/
from langchain_aws import BedrockEmbeddings, BedrockRerank, ChatBedrockConverse
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import DuckDB
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# 設定
model_ids = {
    "embeddings": "amazon.titan-embed-text-v2:0",
    "chat": "anthropic.claude-3-5-sonnet-20241022-v2:0"
}
reranking_model_arn = "arn:aws:bedrock:us-west-2::foundation-model/amazon.rerank-v1:0"
region = "us-west-2"
embeddings_dimension = 1024

# chatClient
chat_client = ChatBedrockConverse(
    model=model_ids["chat"],
    temperature=0.7,
    region_name=region,
    top_p=0.1
)
# embeddings client
embeddings_client = BedrockEmbeddings(
    model_id=model_ids["embeddings"],
    region_name=region
)
# reranking client
reranking_client = BedrockRerank(
    model_arn=reranking_model_arn,
    top_n=10
)

# データをローディング
urls = [
    "https://blogs.nvidia.co.jp/blog/nvidia-expands-omniverse-with-generative-physical-ai/",
    "https://blogs.nvidia.co.jp/blog/nvidia-puts-grace-blackwell-on-every-desk-and-at-every-ai-developers-fingertips/",
    "https://blogs.nvidia.co.jp/blog/nvidia-blackwell-geforce-rtx-50-series-opens-new-world-of-ai-computer-graphics/",
]
 
loader = UnstructuredURLLoader(urls=urls)
# データを読み取る
articles = loader.load()

# チャンキング
splitter = RecursiveCharacterTextSplitter(
    chunk_size=256,
    chunk_overlap=100,
    separators=[
        "\n\n",
        "\n",
        "。",
        "、",
        "",
    ],
    keep_separator='end',
)
chunks = splitter.split_documents(articles)

# vector dbを作成
vectorstore = DuckDB.from_documents(
    documents=chunks,
    embedding=embeddings_client
)

# RAG
TOP_K = 50

base_retriever = vectorstore.as_retriever(search_kwargs={'k': TOP_K})

retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=reranking_client
)

retrieved_chunks = retriever.invoke("NVIDIAの新しく出たGPUには何がありますか？名前を教えてください。")
from pprint import pprint
pprint(retrieved_chunks[0])

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
 
# プロンプトを定義
template = """あなたは優秀なAIアシスタントです。与えられた情報をもとに、ユーザーの質問に回答してください。
## 文脈
{context}
 
## 質問
{input}
"""
prompt = ChatPromptTemplate.from_template(template)
 
# RAGのチェーンを定義
rag_chain = (
  {"context": retriever, "input":RunnablePassthrough()}
  | prompt
  | chat_client
  | StrOutputParser()
)