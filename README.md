# Shrimp Pond Performance Monitoring – Streamlit Application

This repository contains a complete demo solution for the Shrimp Pond Performance Monitoring assignment.  
It provides a clean Streamlit interface, KPI analytics, and an AI assistant using rule-based logic with optional embedding-based retrieval.

## Project Overview

This project analyzes shrimp pond performance using the dataset `data_sample.json`. It includes:

- KPI computation (ABW, DOC, FCR, Survival Rate)
- Growth curve and performance insights
- A rule-based chatbot for answering pond-related questions
- Optional embedding-based search using Gemini embeddings
- A clean and interactive Streamlit dashboard

The solution runs fully offline (except during optional embedding generation).


## Key Features

### 1. KPI Dashboard
- ABW (Average Body Weight)
- DOC (Days of Culture)
- FCR (Feed Conversion Ratio)
- Survival rate analytics
- Growth curve visualization
- Pond-wise performance summary

### 2. Rule-Based AI Assistant
Supports queries such as:
- "Which ponds are harvested?"
- "What is the ABW of pond A1?"
- "Show ponds with high FCR."
- "Provide the summary of pond B2."

### 3. Embedding-Based Retrieval (Optional)
- Uses Gemini `text-embedding-004` (768-dim)
- Saves vectors to `embeddings.pkl`
- Enables semantic similarity search

### 4. Streamlit Interface
- KPI dashboard
- Interactive charts
- Chatbot panel
- Option to force-rebuild embeddings

---

## Repository Structure

SHRIMP_POND_MONITORING/
│── .devcontainer/ # Development container settings
│── README.md # Project documentation
│── app.py # Streamlit UI
│── utils.py # KPI + rule-based QA logic
│── llm_agent.py # Embedding generation + FAISS index
│── data/
│ └── data_sample.json # Dataset
│── embeddings.pkl # Auto-generated embeddings
│── requirements.txt # Dependencies
│── Untitled129.py # Additional script (optional)


## Installation

### 1. Install dependencies
pip install -r requirements.txt

shell
Copy code

### 2. Run the application
streamlit run app.py




### The app loads data from:  
`data/data_sample.json`

## Embedding System 

Embeddings are generated using:

- Model: `text-embedding-004`
- Dimension: 768
- Saved file: `embeddings.pkl`

### Force rebuild embeddings
```python
from llm_agent import build_embeddings
build_embeddings(df, force=True)

### Notes
The AI assistant is rule-based for predictable responses.

Embedding search is optional and enhances semantic retrieval.

Application runs offline unless embeddings are rebuilt.

Intended for demonstration, assignment submission, and prototyping.
### Purpose of This Repository
This project demonstrates:

KPI extraction and trend analysis

Rule-based AI design

Embedding-based search with Gemini

Streamlit UI development

Clean software structure suitable for production-level enhancement
