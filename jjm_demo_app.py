# jjm_demo_app.py
# Unified Streamlit app — Overview layout updated:
# - Left: Scheme Functionality pie
# - Right: Jalmitra Updates pie (today)
# - Below those pies: Top 10 (green) and Worst 10 (red; darkest red for worst)
# - Session-state storage, Plotly 7-day chart, CSV export, etc.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import io
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
st.markdown("Overview: scheme functionality and jalmitra updates side-by-side; Top & Worst lists below them.")
st.markdown("---")

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

def generate_demo_data(total_schemes: int = 20, so_name: str = "SO-Guwahati"):
    """
    Demo generator:
    - Creates schemes (A..), assigns functionality randomly,
    - Maps deterministic Jalmitras JM-1..JM-N to schemes,
    - Creates up to 7 days of readings for Functional schemes using fixed internal probability (85%).
    """
    FIXED_UPDATE_PROB = 0.85  # fixed internal probability
    schemes_rows = []
    jalmitras = [f"JM-{i+1}" for i in range(total_schemes)]
    today = datetime.date.today()
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    # Insert schemes
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

    # Insert readings for functional schemes only
    readings_rows = []
    for idx, row in st.session_state["schemes"].reset_index().iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_id = row["id"]
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
def compute_metrics_and_pivot(readings_df: pd.DataFrame, schemes_df: pd.DataFrame, so_name: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      - last7_all: merged readings for functional schemes in the date window
      - metrics: per-jalmitra metrics (days_updated, total_water_m3, schemes_covered)
    """
    if readings_df.empty or schemes_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    merged = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    mask = (merged['functionality'] == 'Functional') & (merged['so_name'] == so_name) & (merged['reading_date'] >= start_date) & (merged['reading_date'] <= end_date)
    last7_all = merged.loc[mask].copy()

    if last7_all.empty:
        return last7_all, pd.DataFrame()

    metrics = last7_all.groupby('jalmitra').agg(
        days_updated = ('reading_date', lambda x: x.nunique()),
        total_water_m3 = ('water_quantity', 'sum'),
        schemes_covered = ('scheme_id', lambda x: x.nunique())
    ).reset_index()

    metrics['days_updated'] = metrics['days_updated'].astype(int)
    metrics['total_water_m3'] = metrics['total_water_m3'].astype(float).round(2)
    return last7_all, metrics

# ---------------------------
# Demo Data Management UI
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col_gen, col_rem = st.columns([2,1])

with col_gen:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20, step=1)
    if st.button("Generate Demo Data") and not st.session_state.get("generating", False):
        st.session_state["generating"] = True
        with st.spinner("Generating demo data — this may take a few seconds..."):
            generate_demo_data(total_schemes=int(total_schemes))
        st.session_state["generating"] = False
        st.success("✅ Demo data generated in session (fixed internal update probability).")

with col_rem:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("🗑️ All demo data removed from session.")

st.markdown("---")

# ---------------------------
# Role selection (SO / AEE / EE)
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer (AEE) — Placeholder")
    st.info("AEE dashboard placeholder — tell me if you'd like the AEE view to include similar metrics.")
    st.stop()

if role == "Executive Engineer":
    st.header("Executive Engineer (EE) — Placeholder")
    st.info("EE dashboard placeholder — tell me if you'd like the EE view to include district-level aggregation.")
    st.stop()

# ---------------------------
# Section Officer Dashboard (main)
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "SO-Guwahati"

schemes_df = st.session_state["schemes"].copy()
readings_df = st.session_state["readings"].copy()
jalmitras_list = st.session_state["jalmitras"]

if schemes_df.empty:
    st.info("No schemes found. Generate demo data to populate the dashboard.")
    st.stop()

# ---------------------------
# Overview: show an optional image, then side-by-side pies
# ---------------------------
st.subheader("📋 Overview")

# show provided image at top-left of overview if exists
img_path = Path("/mnt/data/assaa.png")
if img_path.exists():
    try:
        st.image(str(img_path), width=220, caption="Overview snapshot")
    except Exception:
        pass

# Prepare data for pies
func_counts = schemes_df['functionality'].value_counts()
# compute today's updates counts (for functional schemes)
today = datetime.date.today().isoformat()
merged_today = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
today_updates = merged_today[
    (merged_today['reading_date'] == today) &
    (merged_today['functionality'] == 'Functional') &
    (merged_today['so_name'] == so_name)
]
updated_count = int(today_updates['jalmitra'].nunique()) if not today_updates.empty else 0
total_functional = int(len(schemes_df[schemes_df['functionality'] == 'Functional']))
absent_count = max(total_functional - updated_count, 0)

# Top row: two pies side-by-side (keep size proportional)
col_left, col_right = st.columns([1,1])

with col_left:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, title="", hole=0.3,
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    fig1.update_traces(textinfo='percent+label')
    # small card-like container: use plotly size via layout margin
    fig1.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220, legend=dict(orientation="h"))
    st.plotly_chart(fig1, use_container_width=True)

with col_right:
    st.markdown("#### Jalmitra Updates (Today)")
    # build df with updated vs absent
    df_part = pd.DataFrame({
        "status": ["Updated", "Absent"],
        "count": [updated_count, absent_count]
    })
    # Ensure counts show 100% when zero rows (avoid empty pie)
    if df_part['count'].sum() == 0:
        # create a neutral pie showing 0/100 for visual parity
        df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[0, total_functional if total_functional>0 else 1]})
    fig2 = px.pie(df_part, names='status', values='count', title="", hole=0.3,
                  color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    fig2.update_traces(textinfo='percent+label')
    fig2.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=220, legend=dict(orientation="h"))
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# ---------------------------
# Below pies: Top & Worst lists (maintain proportions & sizes)
# ---------------------------
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")
start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
end_date = today
last7_all, metrics_cached = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all.empty:
    st.info("No readings in the last 7 days for functional schemes. Generate demo data to see rankings.")
else:
    # ensure all expected jalmitras present
    metrics_df = metrics_cached.copy()
    expected_jalmitras = jalmitras_list if jalmitras_list else [f"JM-{i+1}" for i in range(len(schemes_df))]
    for jm in expected_jalmitras:
        if jm not in metrics_df['jalmitra'].values:
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'jalmitra': jm,
                'days_updated': 0,
                'total_water_m3': 0.0,
                'schemes_covered': 0
            }])], ignore_index=True)

    metrics_df['days_updated'] = metrics_df['days_updated'].astype(int)
    metrics_df['total_water_m3'] = metrics_df['total_water_m3'].astype(float)
    metrics_df['days_norm'] = metrics_df['days_updated'] / 7.0
    max_qty = metrics_df['total_water_m3'].max() if not metrics_df['total_water_m3'].empty else 0.0
    metrics_df['qty_norm'] = metrics_df['total_water_m3'] / max_qty if max_qty > 0 else 0.0

    # fixed 50/50 weights
    weight_freq = 0.5
    weight_qty = 0.5
    metrics_df['score'] = metrics_df['days_norm'] * weight_freq + metrics_df['qty_norm'] * weight_qty

    metrics_df = metrics_df.sort_values(by=['score','total_water_m3'], ascending=False).reset_index(drop=True)
    metrics_df['Rank'] = metrics_df.index + 1
    metrics_df['total_water_m3'] = metrics_df['total_water_m3'].round(2)
    metrics_df['score'] = metrics_df['score'].round(3)

    top_n = 10
    top_table = metrics_df.sort_values(by='score', ascending=False).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
    top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

    # worst_table sorted ascending so worst (lowest score) appears at top
    worst_table = metrics_df.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
    worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

    # Styling: Top = Greens, Worst = dark-red -> light-red
    def style_top(df: pd.DataFrame):
        sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
        sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
        sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
        return sty

    def style_worst(df: pd.DataFrame):
        """
        Use a custom red gradient so the first row (worst) is darkest.
        We'll map values to colors using the 'Reds_r' colormap which is reversed Reds:
        small values (worst scores) -> darker red; larger values -> lighter red.
        """
        sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
        sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds_r')
        sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
        return sty

    # layout: place Top left, Worst right, matching widths and heights
    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        st.dataframe(style_top(top_table), height=420)
        st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode('utf-8'), file_name="jjm_top_10.csv", mime="text/csv")

    with col_w:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.dataframe(style_worst(worst_table), height=420)
        st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode('utf-8'), file_name="jjm_worst_10.csv", mime="text/csv")

st.markdown("---")

# ---------------------------
# 7-day line chart and rest of dashboard unchanged
# ---------------------------
st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")
last7_all2, metrics_cached2 = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all2.empty:
    st.info("No readings for the last 7 days for functional schemes.")
else:
    last_week_qty = last7_all2.groupby(['reading_date','scheme_name'])['water_quantity'].sum().reset_index()
    pivot_chart = last_week_qty.pivot(index='reading_date', columns='scheme_name', values='water_quantity').fillna(0)
    pivot_chart = pivot_chart.sort_index()

    st.markdown("**Chart options**")
    colc1, colc2 = st.columns([2,1])
    with colc1:
        show_total = st.checkbox("Also show total (sum of all functional schemes)", value=True)
        top_k = st.selectbox("Show top N schemes by total water (last 7 days) or 'All'", options=["All","Top 5","Top 10","Top 15"], index=1)
    with colc2:
        date_order = st.radio("Date order", options=["Ascending","Descending"], index=0)

    scheme_sums = last_week_qty.groupby('scheme_name')['water_quantity'].sum().sort_values(ascending=False)
    if top_k == "All":
        selected_schemes = scheme_sums.index.tolist()
    else:
        k = int(top_k.split()[1])
        selected_schemes = scheme_sums.head(k).index.tolist()

    plot_df = last_week_qty[last_week_qty['scheme_name'].isin(selected_schemes)].copy()
    if show_total:
        total_df = last_week_qty.groupby('reading_date')['water_quantity'].sum().reset_index()
        total_df['scheme_name'] = 'Total (all)'
        plot_df = pd.concat([plot_df, total_df], ignore_index=True)

    fig = px.line(plot_df, x='reading_date', y='water_quantity', color='scheme_name',
                  labels={'reading_date': 'Date', 'water_quantity': 'Water (m³)', 'scheme_name': 'Scheme'},
                  markers=True, title="Water Supplied (m³) — last 7 days")
    fig.update_layout(legend_title_text='Scheme / Total')
    if date_order == "Descending":
        fig.update_xaxes(categoryorder='array', categoryarray=sorted(plot_df['reading_date'].unique(), reverse=True))
    st.plotly_chart(fig, use_container_width=True, height=420)

st.markdown("---")
st.subheader("Export Snapshot")
st.markdown("Download current Schemes, Readings, and Metrics as CSVs.")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), file_name='schemes_snapshot.csv', mime='text/csv')
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), file_name='readings_snapshot.csv', mime='text/csv')
# metrics_df might be undefined if no last7_all; guard by using metrics_df if present
try:
    st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False).encode('utf-8'), file_name='metrics_snapshot.csv', mime='text/csv')
except Exception:
    st.info("Metrics CSV not available (no data).")

st.markdown("---")
with st.expander("ℹ️ How ranking is computed"):
    st.markdown("""
    - Days Updated (last 7d): number of distinct days (0–7) a Jalmitra submitted at least one reading.
    - Total Water (m³): cumulative water_quantity for the last 7 days.
    - Normalization:
      - `days_norm = days_updated / 7`
      - `qty_norm = total_water / max_total_water`
    - Score = 0.50 * days_norm + 0.50 * qty_norm (fixed 50/50 weighting).
    - Top table sorts by score descending; Worst table sorts by score ascending.
    """)

st.success(f"Dashboard ready. Demo data generated: {st.session_state.get('demo_generated', False)}. Data stored only for this session.")
