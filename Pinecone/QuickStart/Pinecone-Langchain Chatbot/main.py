from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
from streamlit_chat import message
from utils import *
import streamlit as st

st.subheader("Chatbot with Langchain, ChatGPT, Pinecone, and Streamlit")

if "responses" not in st.session_state:
    st.session_state["responses"] = ["What can I do for you?"]

if "requests" not in st.session_state:
    st.session_state["requests"] = []

llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key="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"
)

if "buffer_memory" not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(
        k=3, return_messages=True
    )

system_msg_template = SystemMessagePromptTemplate.from_template(
    template="""
    Answer the question as truthfully as possible using the provided context, 
    and if the answer is not contained within the text below, say 'I don't know'
    """
)

human_msg_template = HumanMessagePromptTemplate.from_template(
    template="{input}"
)

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template,
     MessagesPlaceholder(variable_name="history"),
     human_msg_template]
)

conversation = ConversationChain(
    memory=st.session_state.buffer_memory,
    prompt=prompt_template,
    llm=llm,
    verbose=True
)

response_container = st.container()
text_container = st.container()

with text_container:
    query = st.text_input("Query: ", key="input")
    if query:
        with st.spinner("Typing"):
            conversation_string = get_conversation_string()
            refined_query = query_refiner(
                conversation=conversation_string,
                query=query
            )
            st.subheader("Refined Query:")
            st.write(refined_query)
            context = find_match(refined_query)
            response = conversation.predict(
                input=f"Context:\n {context} \n\n Query:\n {query}"
            )
        st.session_state["requests"].append(query)
        st.session_state["responses"].append(response)

with response_container:
    if st.session_state["responses"]:
        for i in range(len(st.session_state["responses"])):
            message(st.session_state["responses"][i], key=str(i))
            if i < len(st.session_state["requests"]):
                message(st.session_state["requests"][i],
                        is_user=True, key=str(i) + "_user")
