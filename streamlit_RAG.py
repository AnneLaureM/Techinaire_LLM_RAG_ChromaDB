import streamlit as st
import chromadb
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ChromaDB persistence configuration
PERSIST_DIRECTORY = "RAG_OLLAMA"  # Path where ChromaDB stores data

# Initialize ChromaDB client with persistence
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Load existing collection
try:
    collection = chroma_client.get_collection(name="scientific_corpus")
except Exception as e:
    st.error(f"Error loading collection: {e}")
    collection = None

if collection is None:
    st.stop()

# Function to compute weighted sum of embeddings
def weighted_sum_embeddings(embeddings, weights):
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted_embedding = np.sum(embeddings * weights[:, None], axis=0)
    return weighted_embedding

# Function to compute weighted distance and retrieve the most relevant document
def calculate_weighted_distance(query, collection, title_weight=0.7, text_weight=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    all_documents = collection.get()
    
    if not all_documents or not all_documents['documents'] or not all_documents['metadatas'] or not all_documents['ids']:
        return []

    results = []
    for i in range(len(all_documents['documents'])):
        metadata = all_documents['metadatas'][i] or {}
        title = metadata.get('title', '')
        text = all_documents['documents'][i]
        
        title_embedding = model.encode([title])[0]
        text_embedding = model.encode([text])[0]
        weighted_embedding = weighted_sum_embeddings([title_embedding, text_embedding], [title_weight, text_weight])
        distance = cosine_similarity([query_embedding], [weighted_embedding])[0][0]
        results.append((all_documents['ids'][i], distance))

    results.sort(key=lambda x: x[1], reverse=True)
    return results

# Main function to query Ollama
def query_ollama(query, model_name, title_weight, text_weight, temperature, top_k, top_p):
    results = calculate_weighted_distance(query, collection, title_weight, text_weight)
    
    if results:
        best_match_id = results[0][0]
        fetched_data = collection.get(where={"ids": best_match_id})
        best_match_document = fetched_data['documents'][0] if fetched_data['documents'] else ""
        prompt = f"Context:\n{best_match_document}\n\nQuestion: {query}\nResponse:"
    else:
        prompt = f"⚠️ No relevant context available.\n\nQuestion: {query}\nResponse:"
    
    command = ["ollama", "run", model_name]
    result = subprocess.run(command, input=prompt.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        return f"Error querying Ollama: {result.stderr.decode('utf-8')}"
    
    return result.stdout.decode('utf-8').strip()

# Streamlit UI configuration
st.set_page_config(page_title="RAG Ollama Query", page_icon="🧠", layout="wide")

st.markdown("""
    <style>
    [theme]
    primaryColor="#d33682";
    backgroundColor="#002b36";
    secondaryBackgroundColor="#586e75";
    textColor="#fafafa";
    font="sans serif";
    </style>
""", unsafe_allow_html=True)

st.title("🔍 RAG Ollama Query Interface")

query = st.text_input("Enter your query:", "What is the impact of climate change?")
model_name = st.selectbox("Choose Ollama Model:", ["gemma2:2b", "mistral:7b", "llama2:13b"])
title_weight = st.slider("Title Weight:", 0.0, 1.0, 0.5)
text_weight = st.slider("Text Weight:", 0.0, 1.0, 0.5)
temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
top_k = st.slider("Top K:", 1, 50, 10)
top_p = st.slider("Top P:", 0.0, 1.0, 0.1)

if st.button("🔎 Search"):
    with st.spinner("Processing your query..."):
        response = query_ollama(query, model_name, title_weight, text_weight, temperature, top_k, top_p)
    st.subheader("📌 Response from Ollama:")
    st.write(response)
