"""
RAG + Ollama LLM Pipeline with a REAL Post-Processing Agent (English)
Model: gemma3:4b

What this script does:
1) Loads a persistent ChromaDB collection (scientific_corpus)
2) Retrieves the most relevant document using a weighted embedding of (title, text)
3) Calls Ollama (gemma3:4b) to answer using retrieved context
4) Runs a post-processing AGENT (loop + planning + tools + judging) to improve the answer

Requirements:
- pip install chromadb sentence-transformers scikit-learn numpy
- Ollama installed + `ollama pull gemma3:4b`
- A ChromaDB persistent directory containing collection "scientific_corpus"
"""

import os
import json
import subprocess
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------
# Multiprocessing / env setup
# ---------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    # In some environments, the start method can only be set once.
    pass


# ---------------------------
# Configuration
# ---------------------------
PERSIST_DIRECTORY = "RAG_OLLAMA"
COLLECTION_NAME = "scientific_corpus"
DEFAULT_MODEL = "gemma3:4b"

# Retrieval weights
DEFAULT_TITLE_WEIGHT = 0.5
DEFAULT_TEXT_WEIGHT = 0.5


# ---------------------------
# Ollama helper
# ---------------------------
def ollama_call(prompt: str, model_name: str = DEFAULT_MODEL) -> str:
    """Runs `ollama run <model_name>` and returns stdout."""
    result = subprocess.run(
        ["ollama", "run", model_name],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode("utf-8", errors="ignore"))
    return result.stdout.decode("utf-8", errors="ignore").strip()


def safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """
    Extracts a JSON object from an LLM response that may include extra text or code fences.
    """
    text = text.strip()

    # Strip fenced blocks if present
    candidates: List[str] = []
    if "```" in text:
        parts = text.split("```")
        # Take non-empty parts; often JSON is inside a fenced block
        candidates = [p.strip() for p in parts if p.strip()]
    else:
        candidates = [text]

    for c in candidates:
        c2 = c.strip()
        # Remove leading "json" label if present
        if c2.lower().startswith("json"):
            c2 = c2[4:].strip()

        if "{" in c2 and "}" in c2:
            start = c2.find("{")
            end = c2.rfind("}")
            blob = c2[start : end + 1]
            try:
                return json.loads(blob)
            except Exception:
                continue
    return None


# ---------------------------
# ChromaDB setup
# ---------------------------
def load_collection(persist_dir: str, collection_name: str):
    chroma_client = chromadb.PersistentClient(path=persist_dir)
    try:
        return chroma_client.get_collection(name=collection_name)
    except Exception as e:
        raise ValueError(
            f"Could not load collection '{collection_name}'. "
            f"Make sure it exists in '{persist_dir}'. Error: {e}"
        )


# ---------------------------
# Retrieval (weighted embeddings)
# ---------------------------
def weighted_sum_embeddings(embeddings: List[np.ndarray], weights: List[float]) -> np.ndarray:
    embeddings_arr = np.array(embeddings)
    weights_arr = np.array(weights, dtype=float)
    if weights_arr.sum() == 0:
        weights_arr = np.ones_like(weights_arr)
    weights_arr = weights_arr / weights_arr.sum()
    return np.sum(embeddings_arr * weights_arr[:, None], axis=0)


def calculate_weighted_similarity(
    query: str,
    collection,
    model: SentenceTransformer,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
) -> List[Tuple[str, float]]:
    """
    Returns list of (doc_id, similarity_score) sorted desc.
    """
    query_embedding = model.encode([query])[0]
    all_docs = collection.get()

    if (
        not all_docs
        or not all_docs.get("documents")
        or not all_docs.get("metadatas")
        or not all_docs.get("ids")
    ):
        return []

    results: List[Tuple[str, float]] = []

    for i in range(len(all_docs["documents"])):
        metadata = all_docs["metadatas"][i] or {}
        title = metadata.get("title", "")
        text = all_docs["documents"][i] or ""

        title_emb = model.encode([title])[0] if title else model.encode([""])[0]
        text_emb = model.encode([text])[0] if text else model.encode([""])[0]

        combined = weighted_sum_embeddings([title_emb, text_emb], [title_weight, text_weight])
        sim = float(cosine_similarity([query_embedding], [combined])[0][0])
        results.append((all_docs["ids"][i], sim))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


def retrieve_best_document(
    query: str,
    collection,
    embedder: SentenceTransformer,
    title_weight: float,
    text_weight: float,
) -> Tuple[Optional[str], Optional[Dict[str, Any]], Optional[str], List[Tuple[str, float]]]:
    """
    Returns (doc_text, metadata, doc_id, ranked_results).
    """
    ranked = calculate_weighted_similarity(query, collection, embedder, title_weight, text_weight)
    if not ranked:
        return None, None, None, []

    best_id = ranked[0][0]
    all_docs = collection.get()
    ids = all_docs.get("ids", [])
    docs = all_docs.get("documents", [])
    metas = all_docs.get("metadatas", [])

    if best_id in ids:
        idx = ids.index(best_id)
        doc_text = docs[idx] if idx < len(docs) else None
        meta = metas[idx] if idx < len(metas) else None
        return doc_text, meta, best_id, ranked

    return None, None, best_id, ranked


# ---------------------------
# Post-processing AGENT
# ---------------------------
@dataclass
class PostProcessAgent:
    planner_model: str = DEFAULT_MODEL
    worker_model: str = DEFAULT_MODEL
    judge_model: str = DEFAULT_MODEL
    max_iters: int = 3
    target_score: int = 8  # /10
    history: List[Dict[str, Any]] = field(default_factory=list)

    # --- Tools ---
    def tool_summarize(self, text: str, audience: str) -> str:
        prompt = f"""You are a scientific communication assistant.
Audience: {audience}

Text:
{text}

Task: Summarize in 3-5 short sentences, keep the key points, avoid jargon.
Output:"""
        return ollama_call(prompt, self.worker_model)

    def tool_simplify(self, text: str, audience: str) -> str:
        prompt = f"""You are a scientific communication assistant.
Audience: {audience}

Text:
{text}

Task: Rewrite using simpler words and shorter sentences, keep meaning.
If helpful, add ONE brief analogy.
Output:"""
        return ollama_call(prompt, self.worker_model)

    def tool_rewrite_structured(self, text: str, audience: str) -> str:
        prompt = f"""Rewrite the text to be clear and structured.
Audience: {audience}

Format strictly:
- "Big picture" (1 paragraph)
- "Key points" (3 bullet points)
- "Takeaway" (1 paragraph)

Text:
{text}

Output:"""
        return ollama_call(prompt, self.worker_model)

    def tool_caution_check(self, text: str, question: str) -> str:
        prompt = f"""You are a cautious scientific editor.

Original question:
{question}

Answer:
{text}

Task:
- Identify overconfident statements.
- Rephrase them more cautiously WITHOUT adding new facts.
- Keep it readable.

Return the corrected version:"""
        return ollama_call(prompt, self.worker_model)

    def apply_action(self, action: str, text: str, audience: str, question: str) -> str:
        if action == "summarize":
            return self.tool_summarize(text, audience)
        if action == "simplify":
            return self.tool_simplify(text, audience)
        if action == "rewrite_structured":
            return self.tool_rewrite_structured(text, audience)
        if action == "caution_check":
            return self.tool_caution_check(text, question)
        return text  # no-op fallback

    # --- Planner ---
    def plan_next(self, text: str, audience: str) -> Dict[str, Any]:
        prompt = f"""You are a post-processing agent. Choose ONE action.

Available actions:
- summarize
- simplify
- rewrite_structured
- caution_check
- stop

Heuristics:
- Too long -> summarize
- Too technical -> simplify
- Confusing -> rewrite_structured
- Too certain / too absolute -> caution_check
- Already clear -> stop

Audience: {audience}

Current text:
{text}

Respond ONLY as strict JSON:
{{
  "action": "one_of_actions",
  "reason": "short reason",
  "stop": false
}}"""
        out = ollama_call(prompt, self.planner_model)
        data = safe_json_extract(out)
        if not data or "action" not in data:
            return {"action": "simplify", "reason": "fallback: planner JSON parse failed", "stop": False}

        action = str(data.get("action", "simplify")).strip()
        stop = action == "stop" or bool(data.get("stop", False))
        return {"action": action, "reason": str(data.get("reason", "")).strip(), "stop": stop}

    # --- Judge ---
    def judge(self, text: str, audience: str) -> Tuple[int, str]:
        prompt = f"""You are a quality judge.

Audience: {audience}

Score the text from 0 to 10 based on:
- clarity (0-4)
- simplicity (0-3)
- faithfulness to original meaning (0-3)

Text:
{text}

Respond ONLY as strict JSON:
{{
  "score": 0,
  "feedback": "short feedback"
}}"""
        out = ollama_call(prompt, self.judge_model)
        data = safe_json_extract(out) or {}
        score = data.get("score", 0)
        try:
            score = int(score)
        except Exception:
            score = 0
        feedback = str(data.get("feedback", "")).strip()
        return score, feedback

    # --- Main loop ---
    def run(self, raw_answer: str, question: str, audience: str = "non-expert") -> Dict[str, Any]:
        self.history.clear()
        current = raw_answer

        for step in range(1, self.max_iters + 1):
            score, feedback = self.judge(current, audience)
            self.history.append({"step": step, "type": "judge", "score": score, "feedback": feedback})

            if score >= self.target_score:
                self.history.append({"step": step, "type": "stop", "reason": f"score {score} >= target {self.target_score}"})
                break

            plan = self.plan_next(current, audience)
            self.history.append({"step": step, "type": "plan", **plan})

            if plan.get("stop", False):
                self.history.append({"step": step, "type": "stop", "reason": "planner decided stop"})
                break

            action = plan["action"]
            updated = self.apply_action(action, current, audience, question)
            self.history.append(
                {"step": step, "type": "act", "action": action, "before_len": len(current), "after_len": len(updated)}
            )
            current = updated

        return {"final": current, "history": self.history}


# ---------------------------
# Main RAG pipeline function
# ---------------------------
def rag_answer_with_agent(
    query: str,
    collection,
    embedder: SentenceTransformer,
    model_name: str = DEFAULT_MODEL,
    title_weight: float = DEFAULT_TITLE_WEIGHT,
    text_weight: float = DEFAULT_TEXT_WEIGHT,
    audience: str = "non-expert",
) -> Dict[str, Any]:
    """
    Returns a dict with:
    - retrieved_document
    - retrieved_metadata
    - raw_answer
    - improved_answer
    - agent_history
    - retrieval_ranking (top 10)
    """
    doc_text, meta, best_id, ranked = retrieve_best_document(
        query=query,
        collection=collection,
        embedder=embedder,
        title_weight=title_weight,
        text_weight=text_weight,
    )

    if doc_text:
        prompt = f"""Context:
{doc_text}

Question: {query}

Answer:"""
    else:
        prompt = f"""No context is available.

Question: {query}

Answer:"""

    raw_answer = ollama_call(prompt, model_name=model_name)

    agent = PostProcessAgent(
        planner_model=model_name,
        worker_model=model_name,
        judge_model=model_name,
        max_iters=3,
        target_score=8,
    )
    agent_result = agent.run(raw_answer=raw_answer, question=query, audience=audience)

    return {
        "best_match_id": best_id,
        "retrieved_document": doc_text,
        "retrieved_metadata": meta,
        "raw_answer": raw_answer,
        "improved_answer": agent_result["final"],
        "agent_history": agent_result["history"],
        "retrieval_ranking": ranked[:10],
    }


# ---------------------------
# CLI / Example usage
# ---------------------------
if __name__ == "__main__":
    collection = load_collection(PERSIST_DIRECTORY, COLLECTION_NAME)

    # Load embedder ONCE (important for speed)
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    user_query = "Can you describe the impact of COVID-19?"
    result = rag_answer_with_agent(
        query=user_query,
        collection=collection,
        embedder=embedder,
        model_name=DEFAULT_MODEL,          # gemma3:4b
        title_weight=0.6,
        text_weight=0.4,
        audience="non-expert",
    )

    print("\n====================")
    print("Retrieval")
    print("====================")
    print("Best matched ID:", result["best_match_id"])
    if result["retrieved_metadata"]:
        print("Metadata:", result["retrieved_metadata"])
    if result["retrieved_document"]:
        print("\nRetrieved document (snippet):")
        print(result["retrieved_document"][:800], "..." if len(result["retrieved_document"]) > 800 else "")

    print("\n====================")
    print("Raw LLM Answer")
    print("====================")
    print(result["raw_answer"])

    print("\n====================")
    print("✨ Agent-Improved Answer")
    print("====================")
    print(result["improved_answer"])

    print("\n====================")
    print("🧾 Agent History")
    print("====================")
    for h in result["agent_history"]:
        print(h)
