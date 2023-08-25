from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st

openai.api_key = "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"
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

PINECONE_API_KEY = "bf349a22-3a4e-487a-a05a-14f9193ed5e6"
PINECONE_ENV_KEY = "asia-southeast1-gcp-free"

pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_ENV_KEY
)

index = pinecone.Index("langchain-chatbot")


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(
        queries=input_em,
        top_k=2,
        include_metadata=True
    )
    return result["matches"][0]["metadata"]["text"] + "\n" + \
        result["matches"][1]["metadata"]["text"]


def query_refiner(conversation, query):
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
        temperature=0.5,
        max_tokens=256,
        top_p=1,
        frequency_penalty=0.5,
        presence_penalty=0.5
    )
    return response["choices"][0]["text"]


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state["responses"]) - 1):
        conversation_string += "Human: " + \
            st.session_state["requests"][i] + "\n"
        conversation_string += "Bot: " + \
            st.session_state["responses"][i + 1] + "\n"
    return conversation_string
