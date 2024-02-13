import os
import chromadb
from langchain.vectorstores import Chroma
from langchain.document_transformers import LongContextReorder
from langchain.chains import StuffDocumentsChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers.merger_retriever import MergerRetriever

model_name = "BAAI/bge-small-en"
model_kwargs = {"device": "cpu"}
encode_kwargs = {"normalize_embeddings": True}
hf = HuggingFaceBgeEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
print("Embedding Model Loaded..........")

load_un_sdg_store = Chroma(
    persist_directory="store/un_sdg_chroma_cosine",
    embedding_function=hf
)
print("First Vector Store Loaded.........")

load_paris_agreement_store = Chroma(
    persist_directory="store/paris_chroma_cosine",
    embedding_function=hf
)
print("Second Vector Store Loaded........")

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

for chunks in lotr.get_relevant_documents(
    "Is there any framework available to tackle the climate change?"
):
    print(chunks.page_content)

query = "Is there any framework available to tackle the climate change?"
docs = lotr.get_relevant_documents(query)
print(docs)

reordering = LongContextReorder()
reordered_docs = reordering.transform_documents(docs)
print(reordered_docs)
