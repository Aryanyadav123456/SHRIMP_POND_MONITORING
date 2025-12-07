# llm_agent.py (Final Stable Version - 2025)
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ---------------------------
# Load API Key from gemini.txt
# ---------------------------
if os.path.exists("gemini.txt"):
    with open("gemini.txt", "r") as f:
        key = f.read().strip()

    os.environ["GEMINI_API_KEY"] = key


# ---------------------------
# Constants
# ---------------------------
EMBEDDINGS_FILE = "data/embeddings.pkl"

EMBED_MODEL = "text-embedding-004"     # ALWAYS 768 dimensions
LLM_MODEL = "gemini-2.5-flash"         # Chat model


# ---------------------------
# Configure Gemini
# ---------------------------
genai.configure(api_key=os.environ.get("GEMINI_API_KEY"))


# ---------------------------
# EMBEDDING FUNCTION
# ---------------------------
def embed_text(text: str) -> np.ndarray:
    """
    Returns a 768-d embedding for a given text.
    """
    result = genai.embed_content(
        model=EMBED_MODEL,
        content=text
    )
    emb = np.array(result["embedding"], dtype=float)

    # Enforce correct dimension
    if emb.shape[0] != 768:
        raise ValueError(f"Embedding dimension mismatch: expected 768, got {emb.shape[0]}")

    return emb


# ---------------------------
# BUILD / REBUILD EMBEDDINGS
# ---------------------------
def build_embeddings(df: pd.DataFrame, force=False):
    """
    Builds a lightweight RAG index using 768-dim Gemini embeddings.
    """
    if os.path.exists(EMBEDDINGS_FILE) and not force:
        with open(EMBEDDINGS_FILE, "rb") as f:
            return pickle.load(f)

    texts = (
        "Pond: " + df["pond"].astype(str) +
        ", DOC: " + df["doc"].astype(str) +
        ", ABW: " + df["abw"].astype(str) +
        ", FCR: " + df["fcr"].astype(str) +
        ", Survival: " + df.get("survival_rate", pd.Series([100]*len(df))).astype(str) +
        ", Status: " + df["status"].astype(str)
    )

    vectors = [embed_text(t) for t in texts]

    index = {
        "texts": list(texts),
        "embeddings": np.vstack(vectors),   # shape (N, 768)
        "df_index": df.index.tolist(),
        "dim": 768
    }

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(index, f)

    return index


# ---------------------------
# LOAD SAVED INDEX
# ---------------------------
def load_index():
    if not os.path.exists(EMBEDDINGS_FILE):
        raise FileNotFoundError("No embeddings file found. Build embeddings first.")

    with open(EMBEDDINGS_FILE, "rb") as f:
        return pickle.load(f)


# ---------------------------
# LLM QUERY FUNCTION
# ---------------------------
def answer_with_llm(params: dict):
    query = params["query"]
    index = params["index"]
    top_k = params.get("top_k", 5)
    chat_history = params.get("chat_history", [])

    # ---- 1. Embed query (768-d) ----
    qvec = embed_text(query).reshape(1, -1)

    # ---- 2. Similarity Search ----
    sims = cosine_similarity(qvec, index["embeddings"])[0]
    top_ids = sims.argsort()[-top_k:][::-1]

    retrieved = "\n".join(index["texts"][i] for i in top_ids)

    history_block = "\n".join(
        f"{h['role']}: {h['message']}"
        for h in chat_history[-6:]
    )

    prompt = f"""
You are an expert Shrimp Pond AI Agent.

Use the following retrieved pond data to answer the question.

Retrieved Pond Data:
{retrieved}

Conversation History:
{history_block}

User Question:
{query}

Give:
- Clear answer
- Actionable recommendations
- Short and easy-to-understand explanation
"""

    # ---- 3. Call Gemini LLM ----
    model = genai.GenerativeModel(LLM_MODEL)
    response = model.generate_content(prompt)

    final_answer = response.text

    sources = [
        {"text": index["texts"][i], "score": float(sims[i])}
        for i in top_ids
    ]

    return {
        "answer": final_answer,
        "sources": sources
    }


# ---------------------------
# DIRECT CALL WRAPPER
# ---------------------------
def query_llm(df, query, top_k=5):
    """
    Auto-load index or build if missing.
    """
    if not os.path.exists(EMBEDDINGS_FILE):
        idx = build_embeddings(df, force=True)
    else:
        idx = load_index()

    params = {
        "query": query,
        "index": idx,
        "top_k": top_k,
        "chat_history": []
    }

    return answer_with_llm(params)
