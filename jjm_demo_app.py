# jjm_demo_app.py
# Unified Streamlit app — Overview layout updated (7-day chart removed)
# - Left: Scheme Functionality (Today) pie
# - Right: Jalmitra Updates pie (today)
# - Below those pies: Top 10 (green) and Worst 10 (red; darkest red for worst)
# - Session-state storage, CSV export, etc.
# UPDATED: SO name = "ROKI RAY", Assamese Jalmitra names, Overview shows date & SO name.

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
st.set_page_config(page_title="JJM Unified Dashboard — Overview Layout", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Overview: scheme functionality and Jalmitra updates side-by-side; Top & Worst lists below them.")
st.markdown("---")

# ---------------------------
# Helper: Ensure required columns exist
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
    if "generating" not in st.session_state:
        st.session_state["generating"] = False

init_state()

# ---------------------------
# Utility helpers
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
    st.session_state["generating"] = False

def generate_demo_data(total_schemes: int = 20, so_name: str = "ROKI RAY"):
    """Generate demo schemes and readings with Assamese Jalmitra names."""
    FIXED_UPDATE_PROB = 0.85
    schemes_rows = []

    assamese_names = [
        "Biren", "Nagen", "Rahul", "Vikram", "Debojit", "Anup", "Kamal", "Ranjit", "Himangshu",
        "Pranjal", "Rupam", "Dilip", "Utpal", "Amit", "Jayanta", "Hemanta", "Rituraj", "Dipankar",
        "Bikash", "Dhruba", "Subham", "Pritam", "Saurav", "Bijoy", "Manoj"
    ]
    jalmitras = random.sample(assamese_names * 3, total_schemes)

    today = datetime.date.today()
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    for i in range(total_schemes):
        scheme_name = f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}"
        functionality = random.choice(["Functional", "Non-Functional"])
        scheme_id = st.session_state["next_scheme_id"]
        schemes_rows.append({
            "id": scheme_id,
            "scheme_name": scheme_name,
            "functionality": functionality,
            "so_name": so_name
        })
        st.session_state["next_scheme_id"] += 1

    st.session_state["schemes"] = pd.DataFrame(schemes_rows)
    st.session_state["jalmitras"] = jalmitras

    readings_rows = []
    for idx, row in st.session_state["schemes"].reset_index().iterrows():
        if row.get("functionality", "") != "Functional":
            continue
        scheme_id = row.get("id", None)
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                rid = st.session_state["next_reading_id"]
                readings_rows.append({
                    "id": rid,
                    "scheme_id": scheme_id,
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity": round(random.uniform(40.0, 350.0), 2)
                })
                st.session_state["next_reading_id"] += 1

    st.session_state["readings"] = pd.DataFrame(readings_rows)
    st.session_state["demo_generated"] = True

@st.cache_data
def compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date):
    readings_df = ensure_columns(readings_df.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity"])
    schemes_df = ensure_columns(schemes_df.copy(), ["id","scheme_name","functionality","so_name"])
    if readings_df.empty or schemes_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']],
                               left_on='scheme_id', right_on='id', how='left')
    if 'reading_date' not in merged.columns:
        merged['reading_date'] = ""
    mask = (merged['functionality'] == 'Functional') & (merged['so_name'] == so_name) & \
           (merged['reading_date'] >= start_date) & (merged['reading_date'] <= end_date)
    last7 = merged.loc[mask].copy()

    if last7.empty:
        return last7, pd.DataFrame()

    metrics = last7.groupby('jalmitra').agg(
        days_updated=('reading_date', lambda x: x.nunique()),
        total_water_m3=('water_quantity', 'sum'),
        schemes_covered=('scheme_id', lambda x: x.nunique())
    ).reset_index()

    metrics['days_updated'] = metrics['days_updated'].astype(int)
    metrics['total_water_m3'] = metrics['total_water_m3'].astype(float).round(2)
    return last7, metrics

# ---------------------------
# Demo Data Management UI
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col_gen, col_rem = st.columns([2,1])

with col_gen:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20, step=1)
    if st.button("Generate Demo Data") and not st.session_state.get("generating", False):
        st.session_state["generating"] = True
        with st.spinner("Generating demo data..."):
            generate_demo_data(total_schemes=int(total_schemes))
        st.session_state["generating"] = False
        st.success("✅ Demo data generated for SO: ROKI RAY.")

with col_rem:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("🗑️ All demo data removed from session.")

st.markdown("---")

# ---------------------------
# Role selection
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer (AEE) — Placeholder")
    st.stop()

if role == "Executive Engineer":
    st.header("Executive Engineer (EE) — Placeholder")
    st.stop()

# ---------------------------
# Section Officer Dashboard (main)
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "ROKI RAY"

schemes_df = ensure_columns(st.session_state.get("schemes", pd.DataFrame()).copy(), ["id","scheme_name","functionality","so_name"])
readings_df = ensure_columns(st.session_state.get("readings", pd.DataFrame()).copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity"])
jalmitras_list = st.session_state.get("jalmitras", [])

if schemes_df.empty:
    st.info("No schemes found. Generate demo data first.")
    st.stop()

# ---------------------------
# Overview section (with date and SO name)
# ---------------------------
st.subheader("📋 Overview")

today = datetime.date.today()
today_label = today.strftime("%A, %d %B %Y").upper()
st.markdown(f"**DATE:** {today_label}  **SECTION OFFICER:** {so_name}")

img_path = Path("/mnt/data/assaa.png")
if img_path.exists():
    try:
        st.image(str(img_path), width=220, caption="Overview snapshot")
    except Exception:
        pass

func_counts = schemes_df['functionality'].value_counts()

merged_today = readings_df.merge(
    schemes_df[['id','scheme_name','functionality','so_name']],
    left_on='scheme_id', right_on='id', how='left'
) if not readings_df.empty else pd.DataFrame()
merged_today = ensure_columns(merged_today, ['reading_date','functionality','so_name','jalmitra','scheme_name'])

today_iso = today.isoformat()
today_updates = merged_today[
    (merged_today['reading_date'] == today_iso) &
    (merged_today['functionality'] == 'Functional') &
    (merged_today['so_name'] == so_name)
] if not merged_today.empty else pd.DataFrame()

updated_count = int(today_updates['jalmitra'].nunique()) if not today_updates.empty else 0
total_functional = int(len(schemes_df[schemes_df['functionality'] == 'Functional']))
absent_count = max(total_functional - updated_count, 0)

col_left, col_right = st.columns([1,1])
with col_left:
    st.markdown("#### Scheme Functionality")
    if func_counts.empty:
        func_counts = pd.Series({"Functional":0,"Non-Functional":0})
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, hole=0.3,
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Jalmitra Updates (Today)")
    df_part = pd.DataFrame({"status":["Updated","Absent"],"count":[updated_count,absent_count]})
    if df_part['count'].sum() == 0:
        df_part = pd.DataFrame({"status":["Updated","Absent"],"count":[0,total_functional if total_functional>0 else 1]})
    fig2 = px.pie(df_part, names='status', values='count', hole=0.3,
                  color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------
# Top & Worst Jalmitras
# ---------------------------
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7_all, metrics_cached = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)
metrics_df = pd.DataFrame()

if last7_all.empty:
    st.info("No readings found for last 7 days.")
else:
    metrics_df = metrics_cached.copy()
    expected_jalmitras = jalmitras_list if jalmitras_list else []
    for jm in expected_jalmitras:
        if jm not in metrics_df['jalmitra'].values:
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'jalmitra': jm, 'days_updated': 0, 'total_water_m3': 0.0, 'schemes_covered': 0
            }])], ignore_index=True)

    metrics_df['days_norm'] = metrics_df['days_updated'] / 7.0
    max_qty = metrics_df['total_water_m3'].max() if not metrics_df['total_water_m3'].empty else 0.0
    metrics_df['qty_norm'] = metrics_df['total_water_m3'] / max_qty if max_qty > 0 else 0.0
    metrics_df['score'] = 0.5 * metrics_df['days_norm'] + 0.5 * metrics_df['qty_norm']
    metrics_df = metrics_df.sort_values(by='score', ascending=False).reset_index(drop=True)
    metrics_df['Rank'] = metrics_df.index + 1

    top_table = metrics_df.head(10)[['Rank','jalmitra','days_updated','total_water_m3','score']]
    worst_table = metrics_df.tail(10).sort_values(by='score', ascending=True)[['Rank','jalmitra','days_updated','total_water_m3','score']]

    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        st.dataframe(top_table.style.background_gradient(cmap='Greens', subset=['score']), height=400)

    with col_w:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.dataframe(worst_table.style.background_gradient(cmap='Reds_r', subset=['score']), height=400)

st.markdown("---")

st.subheader("📤 Export Snapshot")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), "schemes_snapshot.csv")
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), "readings_snapshot.csv")
try:
    st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False).encode('utf-8'), "metrics_snapshot.csv")
except Exception:
    st.info("Metrics CSV not available (no data).")

st.success(f"Dashboard ready for SO: {so_name}. Demo data generated: {st.session_state.get('demo_generated', False)} ✅")

