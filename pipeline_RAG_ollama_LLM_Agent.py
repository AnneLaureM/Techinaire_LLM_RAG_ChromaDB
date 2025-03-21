import chromadb
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import os
import multiprocessing as mp

os.environ["TOKENIZERS_PARALLELISM"] = "false"
mp.set_start_method("spawn", force=True)

# Configuration: Path where ChromaDB stores persistent data
PERSIST_DIRECTORY = "RAG_OLLAMA"

# Initialize ChromaDB persistent client
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Load the existing collection
try:
    collection = chroma_client.get_collection(name="scientific_corpus")
except Exception as e:
    print(f"Error loading the collection: {e}")
    collection = None

if collection is None:
    raise ValueError("The collection 'scientific_corpus' does not exist. Please check or create it first.")

# Compute a weighted sum of embeddings
def weighted_sum_embeddings(embeddings, weights):
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted_embedding = np.sum(embeddings * weights[:, None], axis=0)
    return weighted_embedding

# Retrieve the most relevant document based on a weighted similarity
def calculate_weighted_distance(query, collection, title_weight=0.7, text_weight=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    all_documents = collection.get()

    if not all_documents or not all_documents['documents'] or not all_documents['metadatas'] or not all_documents['ids']:
        print("The collection is empty or missing required fields.")
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

# Post-processing agent: summarize, simplify, or rephrase the LLM's response
def explain_answer(answer, model_name="gemma2:2b"):
    prompt = f"""Here is a response generated to a scientific question:

{answer}

Please:
- summarize the answer in a few simple sentences,
- explain the key points in accessible language,
- provide a clearer version for a non-expert audience.

Improved version:"""
    result = subprocess.run(["ollama", "run", model_name],
                            input=prompt.encode('utf-8'),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    if result.returncode != 0:
        print("⚠️ Post-processing agent error:", result.stderr.decode('utf-8'))
        return "Post-processing failed."
    
    return result.stdout.decode('utf-8').strip()

# Main function: query the model using weighted document retrieval
def query_ollama_with_weighted_embedding_and_parameters(query,
                                                        model_name="gemma2:2b",
                                                        title_weight=0.5,
                                                        text_weight=0.5,
                                                        temperature=0.7,
                                                        top_k=10,
                                                        top_p=0.1):

    results = calculate_weighted_distance(query, collection, title_weight, text_weight)

    if results:
        best_match_id = results[0][0]
        print(f"🔍 Best matched ID: {best_match_id}")

        all_docs = collection.get()
        ids = all_docs.get("ids", [])
        documents = all_docs.get("documents", [])
        metadatas = all_docs.get("metadatas", [])

        if best_match_id in ids:
            index = ids.index(best_match_id)
            retrieved_document = documents[index] if index < len(documents) else "No document found"
            retrieved_metadata = metadatas[index] if index < len(metadatas) else "No metadata found"
            
            print(f"📌 Document for ID {best_match_id}:\n{retrieved_document}")
            print(f"🗂 Metadata:\n{retrieved_metadata}")
        else:
            print(f"⚠️ ID {best_match_id} not found in the collection.")
            retrieved_document = "No document found"

        prompt = f"Context:\n{retrieved_document}\n\nQuestion: {query}\n\nAnswer:"
    else:
        print("⚠️ No relevant result found.")
        retrieved_document = None
        prompt = f"No available context.\n\nQuestion: {query}\nAnswer:"

    # Run the base LLM to generate the raw response
    result = subprocess.run(["ollama", "run", model_name],
                            input=prompt.encode('utf-8'),
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Error running Ollama: {result.stderr.decode('utf-8')}")
        return None

    raw_response = result.stdout.decode('utf-8').strip()
    print("\n🧠 Raw Ollama response:\n", raw_response)

    # Run post-processing agent to improve the answer
    improved_response = explain_answer(raw_response)
    print("\n✨ Post-processed response:\n", improved_response)

    return raw_response, improved_response

# Example usage
if __name__ == "__main__":
    user_query = "Can you describe the impact of the covid ?"
    raw, improved = query_ollama_with_weighted_embedding_and_parameters(user_query)
    
    if raw and improved:
        print("\n📜 FINAL RESULTS:")
        print("\n--- RAW RESPONSE ---\n", raw)
        print("\n--- IMPROVED RESPONSE ---\n", improved)
