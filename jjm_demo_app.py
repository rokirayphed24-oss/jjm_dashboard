# jjm_demo_app.py
# Jal Jeevan Mission Dashboard — Locked 50/50 weights
# Now with improved red gradient for Worst Performers table.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import plotly.express as px
from typing import Tuple

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="JJM Dashboard — 50/50 Weighted Ranking", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Ranking uses equal weights: **50% Frequency (Days Updated)** + **50% Quantity (Total Water)**.")
st.markdown("---")

# ---------------------------
# Session state init
# ---------------------------
def init_state():
    if "schemes" not in st.session_state:
        st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    if "readings" not in st.session_state:
        st.session_state["readings"] = pd.DataFrame(columns=[
            "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
        ])
    if "jalmitras" not in st.session_state:
        st.session_state["jalmitras"] = []
    if "next_scheme_id" not in st.session_state:
        st.session_state["next_scheme_id"] = 1
    if "next_reading_id" not in st.session_state:
        st.session_state["next_reading_id"] = 1
    if "demo_generated" not in st.session_state:
        st.session_state["demo_generated"] = False

init_state()

# ---------------------------
# Helper functions
# ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
    ])
    st.session_state["jalmitras"] = []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False

def generate_demo_data(total_schemes: int = 20, so_name: str = "SO-Guwahati"):
    FIXED_UPDATE_PROB = 0.85
    schemes_rows = []
    jalmitras = [f"JM-{i+1}" for i in range(total_schemes)]
    today = datetime.date.today()
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    for i in range(total_schemes):
        scheme_name = f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}"
        functionality = random.choice(["Functional", "Non-Functional"])
        schemes_rows.append({
            "id": st.session_state["next_scheme_id"],
            "scheme_name": scheme_name,
            "functionality": functionality,
            "so_name": so_name
        })
        st.session_state["next_scheme_id"] += 1

    st.session_state["schemes"] = pd.DataFrame(schemes_rows)
    st.session_state["jalmitras"] = jalmitras

    readings_rows = []
    for idx, row in st.session_state["schemes"].iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_id = row["id"]
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                readings_rows.append({
                    "id": st.session_state["next_reading_id"],
                    "scheme_id": scheme_id,
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','30'])}:00",
                    "water_quantity": round(random.uniform(40.0, 350.0), 2)
                })
                st.session_state["next_reading_id"] += 1

    st.session_state["readings"] = pd.DataFrame(readings_rows)
    st.session_state["demo_generated"] = True

@st.cache_data
def compute_metrics(readings_df, schemes_df, so_name, start_date, end_date):
    if readings_df.empty or schemes_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = readings_df.merge(
        schemes_df[["id", "scheme_name", "functionality", "so_name"]],
        left_on="scheme_id", right_on="id", how="left"
    )
    mask = (
        (merged["functionality"] == "Functional") &
        (merged["so_name"] == so_name) &
        (merged["reading_date"] >= start_date) &
        (merged["reading_date"] <= end_date)
    )
    last7 = merged.loc[mask].copy()
    if last7.empty:
        return last7, pd.DataFrame()

    metrics = last7.groupby("jalmitra").agg(
        days_updated=("reading_date", lambda x: x.nunique()),
        total_water_m3=("water_quantity", "sum")
    ).reset_index()
    return last7, metrics

# ---------------------------
# Demo data section
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns(2)

with col1:
    total_schemes = st.number_input("Total demo schemes", 5, 100, 20)
    if st.button("Generate Demo Data"):
        generate_demo_data(total_schemes)
        st.success("✅ Demo data generated successfully!")

with col2:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("🗑️ Demo data cleared successfully!")

st.markdown("---")

# ---------------------------
# Dashboard
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("Currently active for Section Officer role only.")
    st.stop()

st.header("Section Officer Dashboard")
so_name = "SO-Guwahati"
schemes_df = st.session_state["schemes"]
readings_df = st.session_state["readings"]
jalmitras = st.session_state["jalmitras"]

if schemes_df.empty:
    st.info("No data. Generate demo data first.")
    st.stop()

# Pie Chart
func_counts = schemes_df["functionality"].value_counts()
fig_pie = px.pie(
    names=func_counts.index,
    values=func_counts.values,
    title="Scheme Functionality",
    color=func_counts.index,
    color_discrete_map={"Functional": "#4CAF50", "Non-Functional": "#F44336"}
)
st.plotly_chart(fig_pie, use_container_width=False, width=350)

# Today's readings
today = datetime.date.today().isoformat()
st.markdown("---")
st.subheader("📅 Today's Readings (Functional Schemes Only)")

merged_today = readings_df.merge(
    schemes_df[["id", "scheme_name", "functionality", "so_name"]],
    left_on="scheme_id", right_on="id", how="left"
)
readings_today = merged_today[
    (merged_today["reading_date"] == today) &
    (merged_today["functionality"] == "Functional")
][["scheme_name", "jalmitra", "reading", "reading_time", "water_quantity"]]

if readings_today.empty:
    st.info("No readings for today.")
else:
    st.dataframe(readings_today)

# ---------------------------
# 7-day Summary + Rankings
# ---------------------------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance — Last 7 Days")

start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
end_date = today
last7, metrics = compute_metrics(readings_df, schemes_df, so_name, start_date, end_date)

if metrics.empty:
    st.info("No data for last 7 days.")
    st.stop()

# Fill missing Jalmitras with zeros
for jm in jalmitras:
    if jm not in metrics["jalmitra"].values:
        metrics = pd.concat([metrics, pd.DataFrame([{
            "jalmitra": jm,
            "days_updated": 0,
            "total_water_m3": 0.0
        }])], ignore_index=True)

metrics["days_norm"] = metrics["days_updated"] / 7
max_qty = metrics["total_water_m3"].max()
metrics["qty_norm"] = metrics["total_water_m3"] / max_qty if max_qty > 0 else 0
metrics["score"] = 0.5 * metrics["days_norm"] + 0.5 * metrics["qty_norm"]

metrics = metrics.sort_values(by=["score", "total_water_m3"], ascending=False).reset_index(drop=True)
metrics["Rank"] = metrics.index + 1
metrics["score"] = metrics["score"].round(3)

top10 = metrics.head(10).copy()
worst10 = metrics.tail(10).copy().sort_values(by="score", ascending=True)

# ---------------------------
# Styling
# ---------------------------
def style_top(df):
    sty = df.style.format({"total_water_m3": "{:,.2f}", "score": "{:.3f}"})
    sty = sty.background_gradient(subset=["days_updated", "total_water_m3", "score"], cmap="Greens")
    return sty

def style_worst(df):
    sty = df.style.format({"total_water_m3": "{:,.2f}", "score": "{:.3f}"})
    # Use a darker red at the bottom (lowest score)
    sty = sty.background_gradient(
        subset=["days_updated", "total_water_m3", "score"],
        cmap="Reds_r"  # reversed Reds — darkest for worst
    )
    return sty

# ---------------------------
# Display Top & Worst Tables
# ---------------------------
col1, col2 = st.columns(2)
with col1:
    st.markdown("### 🟢 Top 10 Performing Jalmitras")
    st.dataframe(style_top(top10), height=420)

with col2:
    st.markdown("### 🔴 Worst 10 Performing Jalmitras")
    st.dataframe(style_worst(worst10), height=420)

st.markdown("---")
st.markdown("""
**Scoring Formula:**
> Score = 0.5 × (Days Updated / 7) + 0.5 × (Total Water / Max Water)
""")
st.success("Dashboard ready ✅ | Demo data stored only for this session.")
