import streamlit as st
import faiss
import json
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

# Load FAISS index and metadata
@st.cache_resource
def load_index_and_metadata():
    index = faiss.read_index("staxbill_index.faiss")
    with open("staxbill_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    return index, metadata

# Load embedding model
@st.cache_resource
def load_embedder():
    return SentenceTransformer("all-MiniLM-L6-v2")

# Set your OpenAI API key
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
 

# Retrieve context from FAISS index
def retrieve_context(query, index, metadata, embedder, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = "\n\n".join([f"{metadata[i]['title']}\n{metadata[i]['url']}" for i in indices[0]])
    return context

# Generate answer using GPT-4 Turbo
def generate_answer(query, context):
    prompt = f"""
You are a helpful assistant for Stax Bill. Only answer questions that are directly related to Stax Bill's products, services, implementation, or documentation.
If the question is unrelated to Stax Bill, politely respond that you can only assist with Stax Bill-related topics.

Context:
{context}

Question: {query}
Answer:"""

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Stax Bill. Only answer questions related to Stax Bill."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )
    return response.choices[0].message.content

# Streamlit UI
st.set_page_config(page_title="Stax Bill Copilot", layout="centered")
st.title("Stax Bill Copilot Agent")
st.write("Ask a question about Stax Bill")

user_query = st.text_input("🔍 Enter your question:")

if user_query:
    with st.spinner("Retrieving context and generating answer..."):
        index, metadata = load_index_and_metadata()
        embedder = load_embedder()
        context = retrieve_context(user_query, index, metadata, embedder)
        answer = generate_answer(user_query, context)
    st.subheader("🧠 Answer:")
    st.write(answer)
