import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Make environment variables available for application
load_dotenv()

def get_vectorstore_from_url(url):
    # returns a vectorstore from the given website URL
    # get the text in document form
    loader = WebBaseLoader(url)
    documents = loader.load() 
    
    # split document into chunks
    text_splitter = RecursiveCharacterTextSplitter()
    document_chunks = text_splitter.split_documents(documents)
    
    # create a vectorstore from the chunks
    vector_store = Chroma.from_documents(document_chunks, OpenAIEmbeddings())
    
    return vector_store


def get_retriever_chain(vector_store):
    # returns a retriever chain that can be used to retrieve relevant documents based on the chat history and user input
    llm = ChatOpenAI()
    retriever = vector_store.as_retriever()
    
    # create a prompt for the retriever; consisting of the chat history and the user input
    prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        ("user", "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation."),
    ])
    
    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)
    return retriever_chain


def get_conversational_rag_chain(retriever_chain):
    # returns a conversational RAG chain that can be used to generate responses based on the retrieved documents
    llm = ChatOpenAI()
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's qusetion based on the below context:\n\n{context}."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
    ])
    
    stuff_documents_chain  = create_stuff_documents_chain(llm, prompt)
    
    # create a final chain that combines user query, chat history, and retrieved documents
    rag_chain = create_retrieval_chain(retriever_chain, stuff_documents_chain)
    
    return rag_chain
    
    
def get_response(user_input):
    # create conversation chain
    retriever_chain = get_retriever_chain(st.session_state.vector_store)
    
    conversation_rag_chain = get_conversational_rag_chain(retriever_chain)
    
    response = conversation_rag_chain.invoke({
        "chat_history": st.session_state.chat_history,
        "input": user_query
    })
    
    return response['answer']


########################################################### APP ###########################################################
# app config
st.set_page_config(page_title="Chat with Website", page_icon="🤖", layout="wide")
st.title("Chat with Websites")

# sidebar
with st.sidebar:
    st.header("Settings")
    website_url = st.text_input("Website URL")
    
if website_url is None or website_url == "":
    st.info("Please enter a website URL to chat with the website") 
else:
    # Create a Session State so that chat history does NOT reset every time the app refreshes
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            AIMessage(content="Hello, I am Luca 🐶. How can I help you?")
        ] 
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = get_vectorstore_from_url(website_url)

    # Chat with the website
    user_query = st.chat_input("Type your message here...")
    if user_query is not None and user_query != "":
        # Get Response from the AI 
        response = get_response(user_query)
        st.session_state.chat_history.append(HumanMessage(content=user_query))
        st.session_state.chat_history.append(AIMessage(content=response))

    # conversation
    for message in st.session_state.chat_history:
        if isinstance(message, AIMessage):
            with st.chat_message("AI"):
                st.write(message.content)
        elif isinstance(message, HumanMessage):
            with st.chat_message("Human"):
                st.write(message.content)

