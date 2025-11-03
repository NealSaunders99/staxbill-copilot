import faiss
import json
import numpy as np
import openai
from sentence_transformers import SentenceTransformer

# === Load FAISS index and metadata ===
print("📦 Loading FAISS index and metadata...")
index = faiss.read_index("staxbill_index.faiss")
with open("staxbill_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)
print(f"✅ Loaded {len(metadata)} articles.")

# === Load embedding model ===
print("🔍 Loading sentence transformer for semantic search...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === Set your OpenAI API key ===
import os
from dotenv import load_dotenv

load_dotenv()  # Load environment variables from .env

openai.api_key = os.getenv("OPENAI_API_KEY")
``

def retrieve_context(query, top_k=5):
    print(f"\n🔎 Retrieving top {top_k} articles for query: '{query}'")
    query_embedding = embedder.encode([query])
    distances, indices = index.search(np.array(query_embedding), top_k)
    context = "\n\n".join([f"{metadata[i]['title']}\n{metadata[i]['url']}" for i in indices[0]])
    print("📚 Retrieved context:")
    for i in indices[0]:
        print(f" - {metadata[i]['title']}")
    return context

def generate_answer(query, context):
    prompt = f"You are a helpful assistant for Stax Bill. Use the context below to answer the question.\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"
    print("\n🧠 Sending prompt to GPT-4 Turbo...")

    response = openai.chat.completions.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant for Stax Bill."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=500
    )

    answer = response.choices[0].message.content
    print("\n✅ GPT-4 Turbo Response:")
    return answer

# === Run the agent ===
if __name__ == "__main__":
    user_query = input("\n🔍 Ask your Stax Bill question: ")
    context = retrieve_context(user_query)
    answer = generate_answer(user_query, context)
    print("\n🧠 Final Answer:")
    print(answer)