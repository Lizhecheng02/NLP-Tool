from langchain.embeddings import GPT4AllEmbeddings, OllamaEmbeddings
from langchain.vectorstores import chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import WebBaseLoader
import argparse
from langchain.llms import ollama
from langchain import hub
from langchain.chains import RetrievalQA
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

"""
Basic
"""

llm = ollama(
    model="llama2",
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
)

llm("Tell me 5 facts about Roman History:")

"""
Chain
"""

llm = ollama(
    model="llama2",
    temperature=0.9
)

prompt = PromptTemplate(
    input_variables=["topic"],
    template="Give me 5 interesting facts about {topic}."
)

chain = LLMChain(
    llm=llm,
    prompt=prompt,
    verbose=False
)

print(chain.run("the moon"))

"""
RAG
"""


def main():
    parser = argparse.ArgumentParser(
        description="Filter our URL argument."
    )
    parser.add_argument(
        "--url",
        type=str,
        default="http://example.com",
        required=True,
        help="The URL to filter out."
    )

    args = parser.parse_args()
    url = args.url
    print(f"Using URL: {url}")

    loader = WebBaseLoader(url)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=500
    )
    all_splits = text_splitter.split_documents(data)
    print(f"Split into {len(all_splits)} chunks")

    vectorstore = chroma.from_documents(
        documents=all_splits,
        embedding=GPT4AllEmbeddings()
    )
    print(f"Loaded {len(data)} documents")

    QA_CHAIN_PROMPT = hub.pull("rlm/rag-prompt-llama")
    llm = ollama(
        model="llama2",
        verbose=True,
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    )
    print(f"Loaded LLM model {llm.model}")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )

    question = f"What are the latest headlines on {url}"
    result = qa_chain({"query": question})
