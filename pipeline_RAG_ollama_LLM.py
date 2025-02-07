import chromadb
import subprocess
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Configuration de la persistance de ChromaDB
PERSIST_DIRECTORY = "RAG_OLLAMA"  # Chemin où ChromaDB stocke les données

# Initialisation du client ChromaDB avec persistance
chroma_client = chromadb.PersistentClient(path=PERSIST_DIRECTORY)

# Recharger la collection existante (sans la recréer)
try:
    collection = chroma_client.get_collection(name="scientific_corpus")
except Exception as e:
    print(f"Erreur lors du chargement de la collection: {e}")
    collection = None

if collection is None:
    raise ValueError("La collection 'scientific_corpus' n'existe pas. Vérifiez votre stockage ou créez-la d'abord.")

# Fonction pour calculer la somme pondérée des embeddings
def weighted_sum_embeddings(embeddings, weights):
    embeddings = np.array(embeddings)
    weights = np.array(weights)
    weights = weights / weights.sum()
    weighted_embedding = np.sum(embeddings * weights[:, None], axis=0)
    return weighted_embedding

# Fonction pour calculer la distance pondérée et récupérer le document le plus pertinent
def calculate_weighted_distance(query, collection, title_weight=0.7, text_weight=0.3):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])[0]
    all_documents = collection.get()

    if not all_documents or not all_documents['documents'] or not all_documents['metadatas'] or not all_documents['ids']:
        print("La collection est vide ou ne contient pas les données nécessaires.")
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

def query_ollama_with_weighted_embedding_and_parameters(query,
                                                        model_name="gemma2:2b",
                                                        title_weight=0.5,
                                                        text_weight=0.5,
                                                        temperature=0.7,
                                                        top_k=10,
                                                        top_p=0.1):

    results = calculate_weighted_distance(query, collection, title_weight, text_weight)
    print("results : ", results)

    if results:
        best_match_id = results[0][0]
        print(f"🔍 Meilleur ID retourné : {best_match_id}")

        # Récupérer les documents associés à l'ID
        fetched_data = collection.get(where={"ids": best_match_id})
        best_match_document = fetched_data['documents']

        # Affichage pour vérifier la structure
        print("🔍 Données récupérées pour cet ID :", best_match_document)
        prompt = f"Contexte :\n{best_match_document}\n\nQuestion : {query}\n\nRéponse :"


    else:
        print("⚠️ Aucun résultat pertinent trouvé.")
        best_match_document = None
        prompt = f"⚠️ Aucun contexte disponible.\n\nQuestion : {query}\nRéponse :"

    # Construire la commande sans les paramètres invalides
    command = ["ollama", "run", model_name]

    # Exécuter le modèle avec la requête en entrée
    result = subprocess.run(command, input=prompt.encode('utf-8'), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    if result.returncode != 0:
        print(f"Erreur lors de l'interrogation d'Ollama: {result.stderr.decode('utf-8')}")
        return None

    return result.stdout.decode('utf-8').strip()

# Exemple d'utilisation
if __name__ == "__main__":
    user_query = "Can you describe the impact of the covid ?"
    response = query_ollama_with_weighted_embedding_and_parameters(user_query)
    
    if response:
        print("Réponse d'Ollama:\n", response)