import streamlit as st
import uuid
import requests
import json
import chromadb
from chromadb.utils.embedding_functions import OllamaEmbeddingFunction
from chromadb.utils import embedding_functions
import ollama

st.set_page_config(page_title="Ollama RAG Chatbot", page_icon="🤖")

chroma_client = chromadb.PersistentClient(path="ollama")
# openai_ef= ollama.embeddings(model="", prompt="d")
openai_ef = OllamaEmbeddingFunction(
    model_name="mxbai-embed-large",
    url="http://localhost:11434/api/embeddings",
)
collection = chroma_client.get_or_create_collection(
        name="documents",
        embedding_function=openai_ef
    )
# except:
#     collection = chroma_client.get_collection(
#         name="documents",
#         embedding_function=openai_ef
#     )

def generate_embeddings(text):
    return openai_ef([text])[0]

def query_similar_docs(query, n_results=3):
    results = collection.query(
        query_texts=[query],
        n_results=n_results
    )
    return results

def generate_response(prompt, context):
    url = "http://localhost:11434/api/generate"
    enhanced_prompt = f"""Context: {context}\n\nQuestion: {prompt}\n\nAnswer based on the context provided:"""
    
    data = {
        "model": "llama2",
        "prompt": enhanced_prompt,
        "stream": False
    }
    
    try:
        response = requests.post(url, json=data)
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error: {str(e)}"

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("🤖 Ollama RAG Chatbot")

uploaded_file = st.file_uploader("Upload a document", type=["txt"])
if uploaded_file:
    content = uploaded_file.read().decode()
    collection.add(
        documents=[content],
    )
    st.success("Document uploaded and indexed!")

user_input = st.text_input("You:", key="user_input")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    similar_docs = query_similar_docs(user_input)
    context = "\n".join(similar_docs['documents'][0]) if similar_docs['documents'] else ""
    
    bot_response = generate_response(user_input, context)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
    doc_id = str(uuid.uuid4())
    collection.add(ids=[doc_id]  ,documents=[f"""User: {user_input}\nLLM: {bot_response}"""])


for message in st.session_state.messages:
    if message["role"] == "user":
        st.write("You: " + message["content"])
    else:
        st.write("Bot: " + message["content"])
