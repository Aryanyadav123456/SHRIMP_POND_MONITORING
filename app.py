# -------------------------------------------------------------
# app.py — Fixed & Improved
# Shrimp Pond Monitoring – Hybrid AI Chatbot
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import os
import numpy as np
from utils import (
    load_data, preprocess_data, compute_kpis,
    plot_growth_curve, query_ponds_advanced
)
import os
from dotenv import load_dotenv

# Load .env variables
load_dotenv()

# Access the key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

print(GEMINI_API_KEY)  # just to test, remove later

from llm_agent import build_embeddings, answer_with_llm, load_index, EMBEDDINGS_FILE

# ---------------------------
# Helpers: load Gemini API key
# ---------------------------
def load_gemini_key_from_file(path="gemini.txt"):
    if os.path.exists(path):
        with open(path, "r") as f:
            return f.read().strip()
    return None

# Export Gemini key to environment
key = load_gemini_key_from_file()
if key:
    os.environ["GEMINI_API_KEY"] = key

# ---------------------------
# Streamlit config & header
# ---------------------------
st.set_page_config(page_title="🦐 Shrimp Pond Monitoring Agent", layout="wide")
st.title("🦐 Shrimp Pond Performance Monitoring – Hybrid AI Chatbot")
st.caption("Ask anything: growth issues, FCR, survival, pond ranking, harvest status, recommendations…")

DATA_FILE = "data/data_sample.json"

# ---------------------------
# Session state
# ---------------------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "llm_index" not in st.session_state:
    st.session_state.llm_index = None

if "last_llm_sources" not in st.session_state:
    st.session_state.last_llm_sources = []

# ---------------------------
# Load & preprocess data
# ---------------------------
df_raw = load_data(DATA_FILE)
df = preprocess_data(df_raw)

if df.empty:
    st.error("❌ No pond data loaded. Please verify your dataset (data/data_sample.json).")
    st.stop()

# Ensure numeric columns
numeric_cols = ["abw", "weekly_inc", "fcr", "bm", "survival_rate"]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
    else:
        df[col] = 0.0

# Ensure 'day' column exists
if "day" not in df.columns:
    if "sampling_date" in df.columns:
        df["sampling_date"] = pd.to_datetime(df["sampling_date"], errors="coerce")
        df["day"] = df.groupby("pond")["sampling_date"].transform(lambda x: (x - x.min()).dt.days)
    else:
        df["day"] = df.groupby("pond").cumcount() + 1

# ---------------------------
# KPI dashboard
# ---------------------------
st.subheader("📊 Key Performance Indicators")
kpis = compute_kpis(df)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Median ABW (g)", kpis.get("median_abw", 0), help="Average shrimp size")
c2.metric("Median FCR", kpis.get("median_fcr", 0), help="Feed conversion efficiency (lower is better)")
c3.metric("Active Ponds", kpis.get("active_ponds", 0), help="Ponds currently in culture")
c4.metric("Harvested Ponds", kpis.get("harvested_ponds", 0), help="Completed/closed ponds")

# ---------------------------
# Embeddings controls
# ---------------------------
st.sidebar.header("AI / Embeddings")
force_rebuild = st.sidebar.button("Force rebuild embeddings")
show_index_info = st.sidebar.checkbox("Show index info", value=False)

# Build or load embeddings
if st.session_state.llm_index is None or force_rebuild:
    with st.spinner("Building embeddings for fast retrieval (may take a moment)..."):
        try:
            st.session_state.llm_index = build_embeddings(df, force=force_rebuild)
            st.success("Embeddings ready.")
        except Exception as e:
            st.error(f"Failed to build embeddings: {e}")
            st.session_state.llm_index = None

if show_index_info:
    if st.session_state.llm_index:
        idx = st.session_state.llm_index
        st.sidebar.write(f"Index texts: {len(idx['texts'])}")
        st.sidebar.write(f"Embeddings shape: {np.array(idx['embeddings']).shape}")
    else:
        st.sidebar.write("No index built.")

# ---------------------------
# Rule-based -> LLM hybrid
# ---------------------------
def get_actionable_suggestions(row):
    suggestions = []
    try:
        fcr = float(row.get("fcr", 0) or 0)
        survival = float(row.get("survival_rate", 100) or 100)
        doc = float(row.get("doc", 1) or 1)
        abw = float(row.get("abw", 0) or 0)
    except Exception:
        fcr, survival, doc, abw = 0, 100, 1, 0

    if fcr > 1.5:
        suggestions.append("Optimize feed broadcast to reduce FCR.")
    if survival < 85:
        suggestions.append("Inspect water parameters & disease indicators.")
    expected_abw = 0.5 * doc
    if abw < expected_abw:
        suggestions.append("Growth below expectation — review feed schedule or check water quality.")
    return suggestions

def hybrid_query(df, query: str):
    # 1) Rule engine
    rule_res = query_ponds_advanced(df.to_dict(orient="records"), query)

    # 2) LLM fallback
    if rule_res["summary"].startswith("Query not recognized") or not rule_res["data"]:
        if st.session_state.llm_index is None:
            return {"summary": "LLM index not available (embeddings missing).", "data": [], "source": "error"}
        llm_input = {
            "query": query,
            "index": st.session_state.llm_index,
            "top_k": 5,
            "chat_history": st.session_state.chat_history
        }
        try:
            llm_resp = answer_with_llm(llm_input)
            st.session_state.last_llm_sources = llm_resp.get("sources", [])
            return {"summary": llm_resp.get("answer", ""), "data": [], "source": "LLM"}
        except Exception as e:
            return {"summary": f"LLM call failed: {e}", "data": [], "source": "error"}

    # 3) Rule succeeded — enrich, rank, and **preserve all columns**
    df_tmp = pd.DataFrame(rule_res["data"])
    for col in numeric_cols:
        if col not in df_tmp.columns:
            df_tmp[col] = 0.0

    # Preserve all original columns in the dataset
    df_tmp = df_tmp.reindex(columns=df.columns.tolist(), fill_value=np.nan)

    abw_max = df_tmp["abw"].max() or 1.0
    weekly_max = df_tmp["weekly_inc"].max() or 1.0
    fcr_max = df_tmp["fcr"].max() or 1.0
    bm_max = df_tmp["bm"].max() or 1.0

    df_tmp["performance_score"] = (
        0.3 * (df_tmp["abw"] / abw_max) +
        0.3 * (df_tmp["weekly_inc"] / weekly_max) +
        0.2 * (1 - (df_tmp["fcr"] / fcr_max)) +
        0.1 * (df_tmp["bm"] / bm_max) +
        0.1 * (df_tmp["survival_rate"] / 100.0)
    )

    df_tmp = df_tmp.sort_values(by="performance_score", ascending=False)
    out = df_tmp.to_dict(orient="records")
    for r in out:
        r["suggestions"] = get_actionable_suggestions(r)

    return {"summary": rule_res["summary"], "data": out, "source": "rule-based"}

# ---------------------------
# Chat UI
# ---------------------------
st.subheader("💬 Ask Your Question")
col1, col2 = st.columns([4, 1])
with col1:
    user_query = st.text_input("Your question", placeholder="E.g., 'Which ponds are harvested?', 'Top 3 ponds by ABW', 'Why is FCR high?'")
with col2:
    ask_clicked = st.button("Ask", use_container_width=True)

if ask_clicked:
    if not user_query or not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing…"):
            resp = hybrid_query(df, user_query.strip())

        # store history
        st.session_state.chat_history.append({"role": "User", "message": user_query.strip()})
        st.session_state.chat_history.append({"role": resp.get("source", "assistant"), "message": resp.get("summary", "")})

        # show summary
        st.markdown(f"### 📌 Source: **{resp.get('source','unknown')}**")
        st.write(resp.get("summary",""))

        # show LLM sources
        if st.session_state.last_llm_sources:
            st.write("**LLM retrieved contexts:**")
            try:
                st.dataframe(pd.DataFrame(st.session_state.last_llm_sources))
            except Exception:
                st.json(st.session_state.last_llm_sources)

        # show full dataset for rule-based queries
        if resp.get("data"):
            try:
                st.dataframe(pd.DataFrame(resp["data"]))
            except Exception:
                st.json(resp["data"])

# ---------------------------
# Conversation history
# ---------------------------
st.subheader("📝 Conversation History")
for chat in st.session_state.chat_history:
    prefix = "🧑 You:" if chat["role"] == "User" else "🤖"
    st.markdown(f"**{prefix}** {chat['message']}")

# ---------------------------
# Growth curve viewer
# ---------------------------
st.subheader("📈 Growth Curve Viewer")
ponds = ["All"] + sorted(df["pond"].unique().tolist())
selected = st.selectbox("Select pond:", ponds)
if selected == "All":
    plot_growth_curve(df)
else:
    plot_growth_curve(df, pond=selected)

# ---------------------------
# Footer
# ---------------------------
st.markdown("---")
st.caption("Powered by Hybrid Rule-based + Gemini LLM | 🦐 Smart Shrimp Pond Monitoring")
