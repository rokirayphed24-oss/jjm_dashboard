# jjm_demo_app.py
# Unified Streamlit app — full code (only removed: per-day update probability UI & weight slider)
# Behavior:
# - Session-state storage (no SQLite) for stability on Streamlit Cloud
# - Full dashboard: landing, role select (SO/AEE/EE placeholders), demo data generator/remover,
#   schemes lists, today's readings, participation pie, 3-col water table, interactive Plotly 7-day chart,
#   Top (green) & Worst (red) Jalmitra rankings (50/50 weighted score), CSV exports, snapshot exports,
#   deployment notes.
# - Rankings = 50% * days_updated_norm + 50% * qty_norm (locked in code)
# - Worst table uses darkest red for the worst performer and decreases intensity for better ones
#
# Save as jjm_demo_app.py and run: streamlit run jjm_demo_app.py
# Make sure your requirements.txt includes: streamlit, pandas, numpy, plotly

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import io
import plotly.express as px
from typing import Tuple

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="JJM Unified Dashboard (Full)", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Full dashboard. Rankings use a fixed 50% Frequency + 50% Quantity weighting (locked).")
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
    FIXED_UPDATE_PROB = 0.85  # fixed internal probability, removed per-day UI as requested
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

# --- Overview pie chart (Functional vs Non-Functional) ---
if st.session_state.get("demo_generated", False):
    st.subheader("📊 Overview")
    func_counts = schemes_df['functionality'].value_counts()
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, title="Functional vs Non-Functional",
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"}, hole=0.12)
    st.plotly_chart(fig1, use_container_width=False, width=360)

st.markdown("---")
st.subheader("All Schemes under SO")
st.dataframe(schemes_df, height=220)

st.subheader("Functional Schemes under SO")
functional_schemes = schemes_df[schemes_df['functionality'] == "Functional"]
if functional_schemes.empty:
    st.info("No functional schemes found under this SO.")
else:
    st.dataframe(functional_schemes, height=220)

# ---------------------------
# Today's readings (Functional only)
# ---------------------------
today = datetime.date.today().isoformat()
st.markdown("---")
st.subheader("BFM Readings by Jalmitras Today (Functional schemes)")

if readings_df.empty:
    st.info("No readings available. Generate demo data to create readings.")
else:
    merged_today = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    readings_today = merged_today[
        (merged_today['functionality'] == 'Functional') &
        (merged_today['so_name'] == so_name) &
        (merged_today['reading_date'] == today)
    ][['scheme_name','jalmitra','reading','reading_time','water_quantity']]

    st.write(f"Total readings recorded today: **{len(readings_today)}**")
    if not readings_today.empty:
        st.dataframe(readings_today, height=220)
    else:
        st.info("No readings recorded today for functional schemes.")

    # Jalmitra participation pie chart (Updated vs Absent)
    total_functional = len(functional_schemes)
    if total_functional > 0:
        updated_count = readings_today['jalmitra'].nunique() if not readings_today.empty else 0
        absent_count = max(total_functional - updated_count, 0)
        df_part = pd.DataFrame({"status": ["Updated","Absent"], "count": [updated_count, absent_count]})
        fig_part = px.pie(df_part, names='status', values='count', title="Jalmitra Updates Today",
                          color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"}, hole=0.08)
        fig_part.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_part, use_container_width=False, width=360)
    else:
        st.info("No functional schemes — cannot compute participation.")

# Water quantity simple table (3 columns only)
st.markdown("---")
st.subheader("💧 Water Quantity Supplied (m³) per Jalmitra per Scheme (Today)")
if not readings_today.empty:
    simple_table = readings_today[['jalmitra','scheme_name','water_quantity']].copy()
    simple_table.columns = ['Jalmitra','Scheme','Water Quantity (m³)']
    st.dataframe(simple_table, height=220)
    csv_bytes = simple_table.to_csv(index=False).encode('utf-8')
    st.download_button("⬇️ Download Today's Water Table (CSV)", csv_bytes, file_name="water_quantity_today.csv", mime="text/csv")
else:
    st.info("No water quantity measurements to display for today.")

# ---------------------------
# Last 7 Days — Interactive Plotly Line Chart
# ---------------------------
st.markdown("---")
st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")
start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
end_date = today

last7_all, metrics_cached = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all.empty:
    st.info("No data found for the past 7 days.")
else:
    last_week_qty = last7_all.groupby(['reading_date','scheme_name'])['water_quantity'].sum().reset_index()
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

# ---------------------------
# Top & Worst Jalmitra Rankings (weighted 50/50 fixed)
# ---------------------------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")

# Fixed weights 50/50 as requested
weight_freq = 0.5
weight_qty = 0.5

# compute metrics (fallback if cached empty)
if metrics_cached.empty and not last7_all.empty:
    metrics_df = last7_all.groupby('jalmitra').agg(
        days_updated = ('reading_date', lambda x: x.nunique()),
        total_water_m3 = ('water_quantity', 'sum'),
        schemes_covered = ('scheme_id', lambda x: x.nunique())
    ).reset_index()
else:
    metrics_df = metrics_cached.copy()

# Ensure expected jalmitras appear (JM-1..JM-N)
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

# score = weighted average (50/50 locked)
metrics_df['score'] = metrics_df['days_norm'] * weight_freq + metrics_df['qty_norm'] * weight_qty

metrics_df = metrics_df.sort_values(by=['score','total_water_m3'], ascending=False).reset_index(drop=True)
metrics_df['Rank'] = metrics_df.index + 1
metrics_df['total_water_m3'] = metrics_df['total_water_m3'].round(2)
metrics_df['score'] = metrics_df['score'].round(3)

top_n = 10
top_table = metrics_df.sort_values(by='score', ascending=False).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

# worst_table: sort by score ascending so lowest (worst) first
worst_table = metrics_df.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

# ---------------------------
# Styling: Top = Greens, Worst = darkest red for worst → lighter for better
# ---------------------------
def style_top(df: pd.DataFrame):
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

def style_worst(df: pd.DataFrame):
    """
    Use reversed Reds colormap so that lower score (worse) maps to darkest red.
    Since the worst_table is sorted ascending (worst first), we want the first rows
    to appear darkest. background_gradient applies mapping across values, so reversing
    the Reds colormap gives darker color to smaller numeric values.
    """
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds_r')
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

col_left, col_right = st.columns(2)
with col_left:
    st.markdown("### 🟢 Top Performers (Best → Worse)")
    if top_table.empty:
        st.info("No top performers to show.")
    else:
        st.dataframe(style_top(top_table), height=420)
        st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode('utf-8'), file_name="jjm_top_10.csv", mime="text/csv")

with col_right:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to show.")
    else:
        st.dataframe(style_worst(worst_table), height=420)
        st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode('utf-8'), file_name="jjm_worst_10.csv", mime="text/csv")

st.markdown("---")

# Export snapshot (CSV downloads)
st.subheader("Export Snapshot")
st.markdown("Download current Schemes, Readings, and Metrics as CSVs.")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), file_name='schemes_snapshot.csv', mime='text/csv')
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), file_name='readings_snapshot.csv', mime='text/csv')
st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False).encode('utf-8'), file_name='metrics_snapshot.csv', mime='text/csv')

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

# Deployment notes / redirect checklist
st.markdown("---")
st.markdown("### Deployment notes & redirect checklist")
st.markdown("""
If you previously saw a **redirect loop** on Streamlit Cloud, try these checks:
1. Ensure you have no code that manipulates the URL or performs unconditional redirects on load (avoid `st.experimental_set_query_params()` loops).
2. If you use a custom domain, temporarily remove it and confirm the default `*.streamlit.app` URL works.
3. Check **Manage app → Logs** for errors during start-up (dependency install, permission issues).
4. For persistence in production, consider a hosted DB (Supabase/Postgres). Session-state is ephemeral and for demos only.
""")

st.success(f"Dashboard ready. Demo data generated: {st.session_state.get('demo_generated', False)}. Data stored only for this session.")
