import streamlit as st
import chromadb
import subprocess
import numpy as np
import pandas as pd
import json
from datetime import datetime
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "true"
mp.set_start_method("spawn", force=True)


# ------------------------
# ChromaDB Setup
# ------------------------
PERSIST_DIRECTORY = "RAG_OLLAMA"
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

try:
    collection = chroma_client.get_collection(name="scientific_corpus")
except Exception as e:
    st.error(f"Error loading collection: {e}")
    collection = None

if collection is None:
    st.stop()

# ------------------------
# Utility Functions
# ------------------------

def weighted_sum_embeddings(embeddings, weights):
    embeddings = np.array(embeddings)
    weights = np.array(weights) / np.sum(weights)
    return np.sum(embeddings * weights[:, None], axis=0)

def calculate_weighted_distance(query, collection, title_weight=0.7, text_weight=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    all_documents = collection.get()

    if not all_documents or not all_documents['documents']:
        return []

    results = []
    for i, doc in enumerate(all_documents['documents']):
        metadata = all_documents['metadatas'][i] or {}
        title = metadata.get('title', '')
        
        title_embedding = model.encode([title])[0]
        text_embedding = model.encode([doc])[0]
        weighted_embedding = weighted_sum_embeddings([title_embedding, text_embedding], [title_weight, text_weight])
        distance = cosine_similarity([query_embedding], [weighted_embedding])[0][0]
        results.append((all_documents['ids'][i], distance))

    return sorted(results, key=lambda x: x[1], reverse=True)

def explain_answer(answer, model_name):
    prompt = f"""Here is a scientific answer:

{answer}

Please:
- summarize the answer in simple terms,
- explain the key points for a non-expert audience,
- rewrite the answer to make it clearer.

Improved version:"""
    
    result = subprocess.run(["ollama", "run", model_name],
                            input=prompt.encode('utf-8'),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    if result.returncode != 0:
        return f"Error in post-processing: {result.stderr.decode('utf-8')}"
    
    return result.stdout.decode('utf-8').strip()

def query_ollama(query, model_name, title_weight, text_weight, temperature, top_k, top_p):
    results = calculate_weighted_distance(query, collection, title_weight, text_weight)
    
    if results:
        best_match_id = results[0][0]
        all_docs = collection.get()
        ids = all_docs.get("ids", [])
        documents = all_docs.get("documents", [])
        
        if best_match_id in ids:
            index = ids.index(best_match_id)
            retrieved_document = documents[index] if index < len(documents) else "No document found"
        else:
            retrieved_document = "No document found"
    else:
        retrieved_document = "No relevant document found"
    
    prompt = f"Context:\n{retrieved_document}\n\nQuestion: {query}\nResponse:"
    
    command = ["ollama", "run", model_name]
    result = subprocess.run(command, input=prompt.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        return f"Error querying Ollama: {result.stderr.decode('utf-8')}", None

    raw_response = result.stdout.decode('utf-8').strip()
    improved_response = explain_answer(raw_response, model_name)

    return raw_response, improved_response

# ------------------------
# Streamlit UI
# ------------------------

st.set_page_config(page_title="RAG Ollama Query", page_icon="🧠", layout="wide")
st.title("🔍 RAG Ollama Query Interface with Agent & Memory")

# Initialize session state history
if "history" not in st.session_state:
    st.session_state.history = []

# Input UI
query = st.text_input("Enter your query:", "What is the impact of climate change?")
model_name = st.selectbox("Choose LLM Model:", ["gemma2:2b", "mistral:7b", "llama2:13b"])
title_weight = st.slider("Title Weight:", 0.0, 1.0, 0.5)
text_weight = st.slider("Text Weight:", 0.0, 1.0, 0.5)
temperature = st.slider("Temperature:", 0.0, 1.0, 0.7)
top_k = st.slider("Top K:", 1, 50, 10)
top_p = st.slider("Top P:", 0.0, 1.0, 0.1)

# Run Query
if st.button("🔎 Search"):
    with st.spinner("Processing your query..."):
        raw_response, improved_response = query_ollama(query, model_name, title_weight, text_weight, temperature, top_k, top_p)

        if raw_response:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.insert(0, {
                "timestamp": timestamp,
                "query": query,
                "raw": raw_response,
                "improved": improved_response
            })

    if raw_response:
        st.subheader("🧠 Raw Response from Ollama:")
        st.write(raw_response)
    if improved_response:
        st.subheader("✨ Post-processed Response:")
        st.success(improved_response)

# ------------------------
# Sidebar History Tools
# ------------------------
st.sidebar.header("🧠 Query History")

# Export buttons
if st.session_state.history:
    df = pd.DataFrame(st.session_state.history)

    # Export as CSV
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("⬇️ Export History (CSV)", csv, "history.csv", "text/csv")

    # Export as JSON
    json_data = json.dumps(st.session_state.history, indent=2)
    st.sidebar.download_button("⬇️ Export History (JSON)", json_data, "history.json", "application/json")

    # Clear history button
    if st.sidebar.button("🧹 Clear History"):
        st.session_state.history.clear()
        st.sidebar.success("History cleared!")

    # Display history
    for i, item in enumerate(st.session_state.history):
        with st.sidebar.expander(f"{i+1}. {item['timestamp']} - {item['query']}"):
            st.markdown("**Raw response:**")
            st.write(item["raw"])
            st.markdown("**Improved response:**")
            st.success(item["improved"])
else:
    st.sidebar.info("No history yet. Run a query to see it here.")

