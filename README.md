# Shrimp Pond Performance Analysis — Streamlit App (Demo)

This is an end-to-end demo project for the "Shrimp Pond Performance Analysis" interview assignment.
It uses the provided synthetic dataset (`summary_json_aqua_exchange.json`) and implements:
- KPI calculations and dashboards
- A rule-based chatbot that answers pond performance questions
- Streamlit UI for interactive exploration

**How to run**
1. Install requirements: `pip install -r requirements.txt`
2. Run: `streamlit run app.py`
3. The app will load data from `/mnt/data/summary_json_aqua_exchange.json` by default.

**What is inside**:
- `app.py` — Streamlit application (dashboard + chatbot UI)
- `utils.py` — data loaders, KPI functions, and chatbot engine (rule-based)
- `requirements.txt` — Python dependencies
- `data/summary_json_aqua_exchange.json` — dataset (NOT copied if large; app reads from /mnt/data path)
- `README.md` — this file

Notes:
- This is a self-contained, offline demo (no external APIs).
- The chatbot is rule-based and designed to answer common operational queries (current ABW, FCR status, latest sampling, pond summary). It returns a short summary plus a table when multiple records exist.