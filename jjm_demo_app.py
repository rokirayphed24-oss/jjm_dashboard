# jjm_demo_app.py
# JJM Dashboard — styled Top/Worst tables + View charts + Daily BFM Readings Table
# - Adds a "📅 BFM Readings Updated Today" section at bottom
# - Keeps all previous UI elements intact

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px
from typing import Tuple
from pathlib import Path

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="JJM Dashboard — Full Version", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Interactive dashboard for Section Officer **ROKI RAY** — includes Jalmitra performance, daily updates, and readings.")
st.markdown("---")

# ---------------------------
# Utility helpers
# ---------------------------
def ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            if c in ("id", "scheme_id", "reading"):
                df[c] = 0
            elif c == "water_quantity":
                df[c] = 0.0
            else:
                df[c] = ""
    return df

# ---------------------------
# Session state initialization
# ---------------------------
def init_state():
    if "schemes" not in st.session_state:
        st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    if "readings" not in st.session_state:
        st.session_state["readings"] = pd.DataFrame(columns=[
            "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity", "scheme_name"
        ])
    if "jalmitras" not in st.session_state:
        st.session_state["jalmitras"] = []
    if "next_scheme_id" not in st.session_state:
        st.session_state["next_scheme_id"] = 1
    if "next_reading_id" not in st.session_state:
        st.session_state["next_reading_id"] = 1
    if "demo_generated" not in st.session_state:
        st.session_state["demo_generated"] = False
    if "generating" not in st.session_state:
        st.session_state["generating"] = False
    if "selected_jalmitra" not in st.session_state:
        st.session_state["selected_jalmitra"] = None

init_state()

# ---------------------------
# Demo data generation
# ---------------------------
def reset_session_data():
    for key in ["schemes", "readings", "jalmitras"]:
        st.session_state[key] = pd.DataFrame() if key != "jalmitras" else []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False
    st.session_state["selected_jalmitra"] = None

def generate_demo_data(total_schemes: int = 20, so_name: str = "ROKI RAY"):
    FIXED_UPDATE_PROB = 0.85
    schemes_rows, readings_rows = [], []

    assamese_names = [
        "Biren", "Nagen", "Rahul", "Vikram", "Debojit", "Anup", "Kamal", "Ranjit", "Himangshu",
        "Pranjal", "Rupam", "Dilip", "Utpal", "Amit", "Jayanta", "Hemanta", "Rituraj", "Dipankar",
        "Bikash", "Dhruba", "Subham", "Pritam", "Saurav", "Bijoy", "Manoj"
    ]
    jalmitras = random.sample(assamese_names * 3, total_schemes)

    village_names = [
        "Rampur", "Kahikuchi", "Dalgaon", "Guwahati", "Boko", "Moran", "Tezpur", "Sibsagar", "Jorhat", "Hajo",
        "Tihu", "Kokrajhar", "Nalbari", "Barpeta", "Rangia", "Goalpara", "Dhemaji", "Dibrugarh", "Mariani", "Sonari"
    ]
    today = datetime.date.today()
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    for i in range(total_schemes):
        scheme_id = st.session_state["next_scheme_id"]
        schemes_rows.append({
            "id": scheme_id,
            "scheme_name": f"Scheme {chr(65+i)}",
            "functionality": random.choice(["Functional", "Non-Functional"]),
            "so_name": so_name
        })
        st.session_state["next_scheme_id"] += 1

    st.session_state["schemes"] = pd.DataFrame(schemes_rows)
    st.session_state["jalmitras"] = jalmitras

    for idx, row in st.session_state["schemes"].iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_label = random.choice(village_names) + " PWSS"
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                readings_rows.append({
                    "id": st.session_state["next_reading_id"],
                    "scheme_id": row["id"],
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity": round(random.uniform(40.0, 350.0), 2),
                    "scheme_name": scheme_label
                })
                st.session_state["next_reading_id"] += 1

    st.session_state["readings"] = pd.DataFrame(readings_rows)
    st.session_state["demo_generated"] = True

# ---------------------------
# Metric computation
# ---------------------------
@st.cache_data
def compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date):
    readings_df = ensure_columns(readings_df.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    schemes_df = ensure_columns(schemes_df.copy(), ["id","scheme_name","functionality","so_name"])
    merged = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']],
                               left_on='scheme_id', right_on='id', how='left')
    mask = (merged['functionality'] == 'Functional') & (merged['so_name'] == so_name) & \
           (merged['reading_date'] >= start_date) & (merged['reading_date'] <= end_date)
    last7 = merged.loc[mask].copy()
    if last7.empty:
        return last7, pd.DataFrame()
    metrics = last7.groupby('jalmitra').agg(
        days_updated=('reading_date', lambda x: x.nunique()),
        total_water_m3=('water_quantity', 'sum')
    ).reset_index()
    return last7, metrics

# ---------------------------
# UI — Demo data management
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col_gen, col_rem = st.columns([2,1])
with col_gen:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=100, value=20)
    if st.button("Generate Demo Data"):
        generate_demo_data(total_schemes)
        st.success("✅ Demo data generated for SO: ROKI RAY.")
with col_rem:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("🗑️ All demo data removed.")
st.markdown("---")

# ---------------------------
# Role Selection
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.header(f"{role} Dashboard — Placeholder")
    st.stop()

# ---------------------------
# Overview Section
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "ROKI RAY"
today = datetime.date.today()
st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}  **SECTION OFFICER:** {so_name}")

schemes_df = st.session_state["schemes"]
readings_df = st.session_state["readings"]

if schemes_df.empty:
    st.info("No schemes found. Generate demo data.")
    st.stop()

# Functionality pie + Updates pie
func_counts = schemes_df['functionality'].value_counts()
today_iso = today.isoformat()
merged_today = readings_df.merge(
    schemes_df[['id','scheme_name','functionality','so_name']],
    left_on='scheme_id', right_on='id', how='left'
)
today_updates = merged_today[
    (merged_today['reading_date'] == today_iso) &
    (merged_today['functionality'] == 'Functional') &
    (merged_today['so_name'] == so_name)
]
updated_count = today_updates['jalmitra'].nunique()
total_functional = len(schemes_df[schemes_df['functionality']=='Functional'])
absent_count = max(total_functional - updated_count, 0)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                  color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.markdown("#### Jalmitra Updates (Today)")
    df_part = pd.DataFrame({"status":["Updated","Absent"],"count":[updated_count,absent_count]})
    fig2 = px.pie(df_part, names='status', values='count',
                  color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------
# Performance tables + View feature (same as before)
# ---------------------------
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7_all, metrics = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all.empty:
    st.info("No readings found for the last 7 days.")
else:
    metrics['score'] = 0.5*(metrics['days_updated']/7) + 0.5*(metrics['total_water_m3']/metrics['total_water_m3'].max())
    metrics = metrics.sort_values(by='score', ascending=False).reset_index(drop=True)
    metrics['Rank'] = metrics.index + 1
    village_names = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    metrics['Scheme Name'] = [random.choice(village_names)+" PWSS" for _ in range(len(metrics))]
    top10 = metrics.head(10).copy()
    worst10 = metrics.tail(10).sort_values(by='score').copy()

    colt, colw = st.columns(2)
    with colt:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        st.dataframe(top10.style.background_gradient(cmap='Greens', subset=['days_updated','total_water_m3','score']))
    with colw:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.dataframe(worst10.style.background_gradient(cmap='Reds_r', subset=['days_updated','total_water_m3','score']))

# ---------------------------
# NEW SECTION — Daily BFM Readings
# ---------------------------
st.markdown("---")
st.subheader("📅 BFM Readings Updated Today")

if today_updates.empty:
    st.info("No BFM readings recorded today.")
else:
    daily_bfm = today_updates[['jalmitra','scheme_name','reading','water_quantity']].copy()
    daily_bfm.columns = ['Jalmitra','Scheme Name','BFM Reading','Water Quantity (m³)']
    daily_bfm = daily_bfm.sort_values('Jalmitra')
    st.dataframe(
        daily_bfm.style.format({'BFM Reading':'{:06d}','Water Quantity (m³)':'{:.2f}'}).background_gradient(
            cmap='Blues', subset=['BFM Reading','Water Quantity (m³)']),
        height=350
    )

# ---------------------------
# Export Section
# ---------------------------
st.markdown("---")
st.subheader("📤 Export Snapshot")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), "schemes_snapshot.csv")
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), "readings_snapshot.csv")
st.download_button("Download Metrics CSV", metrics.to_csv(index=False).encode('utf-8'), "metrics_snapshot.csv")
st.success(f"Dashboard ready for SO: {so_name}. ✅ Demo data generated: {st.session_state['demo_generated']}")
