Shrimp Pond Performance Monitoring – Streamlit Application

This repository contains a complete demo solution for the Shrimp Pond Performance Monitoring assignment.
It provides a clean Streamlit interface, KPI analytics, and a simple AI assistant using rule-based logic and optional embedding-based retrieval.

Project Overview

This project is designed to analyze shrimp pond performance using the provided dataset data_sample.json.
It includes:

KPI computation (ABW, DOC, FCR, Survival Rate)

Growth curve and performance insights

A rule-based chatbot for answering pond-related questions

Optional embedding-based search using Gemini embeddings

A clean and interactive Streamlit dashboard

The solution is fully local and works offline (except for optional embedding generation).

Key Features
1. KPI Dashboard

Current ABW

DOC trends

FCR monitoring

Survival rate (if available)

Growth curve visualization

Pond-wise performance summary

2. Rule-Based AI Assistant

Answers questions such as:

“Which ponds are harvested?”

“What is the ABW of pond A1?”

“Show ponds with high FCR.”

“Give the summary of pond B2.”

3. Embedding-Based Retrieval (Optional)

Uses Gemini text-embedding-004 (768-dim)

Embeddings stored in embeddings.pkl

Used for semantic similarity search (not for LLM generation)

4. Streamlit UI

Dashboard section

Chat section

Option to rebuild embeddings

Completely browser-based

Repository Structure
SHRIMP_POND_MONITORING/
│── .devcontainer/              # Development container files
│── README.md                   # Project documentation
│── Untitled129.py              # Supporting script (optional/testing)
│── app.py                      # Main Streamlit app
│── data/                       # Data folder
│     └── data_sample.json      # Dataset used by the app
│── embeddings.pkl              # Saved embeddings (auto-generated)
│── llm_agent.py                # Embedding logic + semantic search
│── requirements.txt            # Project dependencies
│── utils.py                    # KPI, rule-based QA, helper functions

Installation
1. Install dependencies
pip install -r requirements.txt

2. Run the application
streamlit run app.py


The app automatically loads the dataset from:

data/data_sample.json

Embedding System (Optional)

Embeddings are generated using:

Model: text-embedding-004

Dimension: 768

Saved to: embeddings.pkl

To force rebuild embeddings:

from llm_agent import build_embeddings
build_embeddings(df, force=True)


To load saved embeddings:

from llm_agent import load_index
index = load_index()


This system is used to improve similarity matching inside the chatbot.

Notes

The chatbot is intentionally rule-based, ensuring deterministic outputs.

Embedding-based search is optional and used only for semantic matching, not for LLM queries.

The app does not require an internet connection unless you regenerate embeddings.

Purpose of This Submission

This project demonstrates:

Data extraction and transformation

KPI and trend analysis

Simple AI-driven insights

Clean UI/UX using Streamlit

Use of embeddings for semantic matching
