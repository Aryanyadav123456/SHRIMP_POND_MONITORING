# utils.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Dict, Any
import plotly.express as px
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

# ------------------------------
# Load pond data
# ------------------------------
def load_data(file_path: str = "data/data_sample.json") -> pd.DataFrame:
    """
    Load pond data from CSV or JSON with fallback.
    Handles JSON with top-level 'data' key.
    """
    try:
        if file_path.endswith(".json"):
            raw = pd.read_json(file_path)
            if "data" in raw.columns:
                df = pd.DataFrame(raw["data"].tolist())
            else:
                df = raw
        else:
            df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        return pd.DataFrame()


# ------------------------------
# Preprocess Data
# ------------------------------
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean columns, convert dates, standardize pond names, and derive 'day'.
    """
    if df.empty:
        return df

    # Standardize columns
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Convert date columns
    if "sampling_date" in df.columns:
        df["sampling_date"] = pd.to_datetime(df["sampling_date"], format="%d-%m-%Y", errors='coerce')
    if "stocking_date" in df.columns:
        df["stocking_date"] = pd.to_datetime(df["stocking_date"], format="%d-%m-%Y", errors='coerce')

    # Ensure 'pond' column exists
    if "pond" not in df.columns:
        possible_ponds = [c for c in df.columns if "pond" in c]
        if possible_ponds:
            df = df.rename(columns={possible_ponds[0]: "pond"})
        else:
            df["pond"] = "Unknown"

    # Derive 'day' as days since stocking
    if "stocking_date" in df.columns and "sampling_date" in df.columns:
        df["day"] = (df["sampling_date"] - df["stocking_date"]).dt.days

    return df


# ------------------------------
# Compute KPIs
# ------------------------------
def compute_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute key performance indicators with safe fallbacks.
    """
    kpis = {}
    kpis['median_abw'] = df['abw'].median() if 'abw' in df.columns and not df['abw'].empty else 0
    kpis['median_fcr'] = df['fcr'].median() if 'fcr' in df.columns and not df['fcr'].empty else 0
    kpis['active_ponds'] = df[df.get('status', '').str.upper() == "ACTIVE"]['pond'].nunique() if 'status' in df.columns else 0
    kpis['harvested_ponds'] = df[df.get('status', '').str.upper() == "HARVESTED"]['pond'].nunique() if 'status' in df.columns else 0
    return kpis


# ------------------------------
# Plot Growth Curve
# ------------------------------
def plot_growth_curve(df: pd.DataFrame, pond: str = None):
    """
    Plot ABW and FCR growth curves per pond with fallback.
    Automatically uses 'day' or 'sampling_date' for x-axis.
    """
    if df.empty:
        st.warning("No data available to plot.")
        return

    data = df if pond is None else df[df['pond'] == pond]

    # Choose x-axis
    if 'day' in data.columns:
        x_col = 'day'
    elif 'sampling_date' in data.columns:
        x_col = 'sampling_date'
    else:
        st.warning("No 'day' or 'sampling_date' column found for growth curve plotting.")
        return

    metrics = ['abw', 'fcr']
    for metric in metrics:
        if metric in data.columns:
            fig = px.line(
                data,
                x=x_col,
                y=metric,
                color='pond' if 'pond' in data.columns else None,
                title=f'{metric.upper()} Growth Curve'
            )
            st.plotly_chart(fig)


# ------------------------------
# Advanced Rule-based Chatbot
# ------------------------------
def query_ponds_advanced(data: List[Dict[str, Any]], q: str, top_n: int = 5) -> Dict[str, Any]:
    """
    Advanced pond query with fallbacks for missing columns, empty datasets,
    time-series trends, harvest estimation, and forecast.
    """
    if not data:
        return {"summary": "No pond data provided.", "data": []}

    df = pd.DataFrame(data)
    if df.empty:
        return {"summary": "Data is empty.", "data": []}

    q_lower = q.lower()

    def has_col(col: str) -> bool:
        return col in df.columns and not df[col].empty

    # ---------- STATUS QUERIES ----------
    if "harvest" in q_lower:
        if has_col("status"):
            harvested = df[df["status"].str.upper() == "HARVESTED"]
            ponds = harvested["pond"].unique().tolist()
            summary = f"Harvested ponds: {', '.join(map(str, ponds))}" if ponds else "No harvested ponds found."
            return {"summary": summary, "data": harvested.to_dict(orient="records")}
        return {"summary": "No 'status' column available.", "data": []}

    if "active" in q_lower:
        if has_col("status"):
            active = df[df["status"].str.upper() == "ACTIVE"]
            ponds = active["pond"].unique().tolist()
            summary = f"Active ponds: {', '.join(map(str, ponds))}" if ponds else "No active ponds found."
            return {"summary": summary, "data": active.to_dict(orient="records")}
        return {"summary": "No 'status' column available.", "data": []}

    # ---------- METRIC QUERIES ----------
    metrics = {
        "abw": "g",
        "fcr": "",
        "doc": "mg/L",
        "biomass": "kg",
        "survival_rate": "%"
    }

    for metric, unit in metrics.items():
        if metric in q_lower:
            if has_col(metric):
                # Top N ponds
                if "top" in q_lower or "highest" in q_lower:
                    top_df = df.sort_values(metric, ascending=False).head(top_n)
                    top_ponds = ', '.join(top_df['pond'].astype(str).tolist())
                    summary = f"Top {top_n} ponds by {metric.upper()}: {top_ponds}"
                    return {"summary": summary, "data": top_df.to_dict(orient="records")}

                # Bottom N ponds
                if "bottom" in q_lower or "lowest" in q_lower:
                    bottom_df = df.sort_values(metric, ascending=True).head(top_n)
                    bottom_ponds = ', '.join(bottom_df['pond'].astype(str).tolist())
                    summary = f"Bottom {top_n} ponds by {metric.upper()}: {bottom_ponds}"
                    return {"summary": summary, "data": bottom_df.to_dict(orient="records")}

                # Time-series trends
                if "trend" in q_lower or "time-series" in q_lower:
                    if not has_col("day"):
                        return {"summary": "No 'day' column for time-series analysis.", "data": []}
                    trends = {}
                    for pond in df["pond"].unique():
                        pond_df = df[df["pond"] == pond].sort_values("day")
                        trends[pond] = pond_df[["day", metric]].to_dict(orient="records")
                    return {"summary": f"Time-series trends for {metric.upper()} generated per pond.", "data": trends}

                # Harvest date estimation
                if "harvest date" in q_lower or "estimate harvest" in q_lower:
                    if metric != "abw" or not has_col("day"):
                        return {"summary": "Harvest estimation requires 'ABW' and 'day' columns.", "data": []}
                    harvest_estimates = {}
                    target_abw = 20  # grams
                    for pond in df["pond"].unique():
                        pond_df = df[df["pond"] == pond].sort_values("day")
                        if pond_df.empty:
                            harvest_estimates[pond] = "No data"
                            continue
                        try:
                            X = pond_df["day"].values.reshape(-1, 1)
                            y = pond_df["abw"].values
                            model = LinearRegression().fit(X, y)
                            est_day = (target_abw - model.intercept_) / model.coef_[0]
                            harvest_estimates[pond] = round(float(est_day), 1)
                        except Exception:
                            harvest_estimates[pond] = "Estimation failed"
                    return {"summary": f"Estimated harvest days for ABW={target_abw}g per pond.", "data": harvest_estimates}

                # Default stats
                median_val = df[metric].median()
                max_val = df[metric].max()
                min_val = df[metric].min()
                summary = f"{metric.upper()} stats — median: {median_val:.2f}{unit}, max: {max_val:.2f}{unit}, min: {min_val:.2f}{unit}"
                if metric in ["biomass", "survival_rate"] and has_col("pond"):
                    max_pond = df.loc[df[metric] == max_val, "pond"].iloc[0]
                    summary += f" (Highest in pond: {max_pond})"
                return {"summary": summary, "data": []}
            return {"summary": f"No '{metric.upper()}' column found.", "data": []}

    # ---------- FORECASTING ----------
    if "forecast" in q_lower:
        forecast_metric = None
        for m in metrics.keys():
            if m in q_lower:
                forecast_metric = m
                break
        if forecast_metric is None or not has_col(forecast_metric) or not has_col("day"):
            return {"summary": "Forecast requires 'day' column and a valid metric (ABW, FCR, biomass, survival_rate).", "data": []}

        forecasts = {}
        for pond in df["pond"].unique():
            pond_df = df[df["pond"] == pond].sort_values("day")
            if pond_df.empty:
                forecasts[pond] = "No data"
                continue
            try:
                X = pond_df["day"].values.reshape(-1, 1)
                y = pond_df[forecast_metric].values
                model = LinearRegression().fit(X, y)
                future_days = np.arange(pond_df["day"].max() + 1, pond_df["day"].max() + 8).reshape(-1, 1)
                y_pred = model.predict(future_days)
                forecasts[pond] = [{"day": int(future_days[i][0]), forecast_metric: round(float(y_pred[i]), 2)} for i in range(7)]
            except Exception:
                forecasts[pond] = "Forecast failed"
        return {"summary": f"7-day forecast for {forecast_metric.upper()} per pond generated.", "data": forecasts}

    # Default fallback
    return {"summary": "Query not recognized. Try: harvest, active, ABW, FCR, DOC, survival_rate, biomass, trend, harvest date, forecast.", "data": []}
