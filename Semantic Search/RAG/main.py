import os
import chromadb
from langchain.retrievers.merger_retriever import MergerRetriever
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.document_transformers import EmbeddingsRedundantFilter, EmbeddingsClusteringFilter
from langchain.retrievers.document_compressors import DocumentCompressorPipeline
from langchain.retrievers import ContextualCompressionRetriever
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Model Loaded..........")

load_un_sdg = PyPDFLoader("data/UN SDG.pdf")
documents_un_sdg = load_un_sdg.load()
text_splitter_un_sdg = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)
texts_un_sdg = text_splitter_un_sdg.split_documents(
    documents=documents_un_sdg
)

load_paris_agreement = PyPDFLoader("data/english_paris_agreement.pdf")
documents_paris_agreement = load_paris_agreement.load()
text_splitter_paris_agreement = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=100
)
texts_paris_agreement = text_splitter_paris_agreement.split_documents(
    documents=documents_paris_agreement
)

print(texts_paris_agreement[0])

un_sdg_store = Chroma.from_documents(
    texts_un_sdg,
    hf,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="store/un_sdg_chroma_cosine"
)

paris_agreement_store = Chroma.from_documents(
    texts_paris_agreement,
    hf,
    collection_metadata={"hnsw:space": "cosine"},
    persist_directory="store/paris_chroma_cosine"
)

load_un_sdg_store = Chroma(
    persist_directory="store/un_sdg_chroma_cosine",
    embedding_function=hf
)

load_paris_agreement_store = Chroma(
    persist_directory="store/paris_chroma_cosine",
    embedding_function=hf
)

retriever_un_sdg = load_un_sdg_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "include_metadata": True
    }
)

retriever_paris_agreement = load_paris_agreement_store.as_retriever(
    search_type="similarity",
    search_kwargs={
        "k": 3,
        "include_metadata": True
    }
)

lotr = MergerRetriever(
    retrievers=[retriever_un_sdg, retriever_paris_agreement]
)
print(lotr)

for chunks in lotr.get_relevant_documents(
    "Is there any framework available to tackle the climate change?"
):
    print(chunks.page_content)
