# jjm_demo_app.py
# Unified Streamlit app with Plotly (session-state) — Weighted ranking restored
# - Ranking = weighted average of (days updated) and (total water)
# - Single slider: Frequency weight (%) — default 50%. Quantity weight = 100 - freq.
# - Session-state storage, Plotly chart, Top/Worst tables, CSV export.

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
st.set_page_config(page_title="JJM Unified Dashboard (Weighted Ranking)", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Ranking uses a weighted average of days-updated and total water. Adjust the Frequency weight below.")
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
    if "generating" not in st.session_state:
        st.session_state["generating"] = False

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
    st.session_state["generating"] = False

def generate_demo_data(total_schemes: int = 20, so_name: str = "SO-Guwahati"):
    """
    Populate st.session_state with demo schemes and readings (functional schemes only).
    Uses a fixed internal per-day update probability (85%).
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
    Compute:
     - last7_all: merged readings for functional schemes in date window
     - metrics: per-jalmitra metrics (days_updated, total_water_m3, schemes_covered)
    Returns (last7_all, metrics)
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
# Demo data management UI
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col_g, col_r = st.columns([2,1])

with col_g:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20, step=1)
    if st.button("Generate Demo Data") and not st.session_state.get("generating", False):
        st.session_state["generating"] = True
        with st.spinner("Generating demo data — this may take a few seconds..."):
            generate_demo_data(total_schemes=int(total_schemes))
        st.session_state["generating"] = False
        st.success("Demo data generated in session (fixed internal update probability).")

with col_r:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("Session demo data cleared.")

st.markdown("---")

# ---------------------------
# Role selection & placeholders
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer (AEE) — Placeholder")
    st.info("AEE view is a placeholder. Tell me if you want the AEE view adapted.")
    st.stop()
if role == "Executive Engineer":
    st.header("Executive Engineer (EE) — Placeholder")
    st.info("EE view is a placeholder. Tell me if you want the EE view adapted.")
    st.stop()

# ---------------------------
# Section Officer Dashboard
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "SO-Guwahati"

schemes_df = st.session_state['schemes'].copy()
readings_df = st.session_state['readings'].copy()
jalmitras_list = st.session_state['jalmitras']

if schemes_df.empty:
    st.info("No schemes found in this session. Use 'Generate Demo Data' to populate the dashboard.")
    st.stop()

# Overview pie (functional vs non-functional)
if st.session_state.get("demo_generated", False):
    st.subheader("📊 Scheme Functionality")
    func_counts = schemes_df['functionality'].value_counts()
    fig_pie = px.pie(names=func_counts.index, values=func_counts.values, title="Functional vs Non-Functional", hole=0.15,
                     color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    st.plotly_chart(fig_pie, use_container_width=False, width=360)

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
    st.info("No readings found. Generate demo data to create example readings.")
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
        df_part = pd.DataFrame({
            "status": ["Updated", "Absent"],
            "count": [updated_count, absent_count]
        })
        fig_part = px.pie(df_part, names='status', values='count', title="Jalmitra Updates Today",
                          color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"},
                          hole=0.08)
        fig_part.update_traces(textinfo='percent+label')
        st.plotly_chart(fig_part, use_container_width=False, width=360)
    else:
        st.info("No functional schemes — cannot compute participation.")

# Water quantity simple table (3 columns)
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
    st.info("No readings for the last 7 days for functional schemes.")
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
# Top / Worst Jalmitra Rankings (Last 7 days) — weighted scoring
# ---------------------------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance Rankings (Last 7 Days) — Weighted Score")

# Slider to control frequency weight (0-100); quantity weight = 100 - freq
freq_weight_pct = st.slider("Frequency weight (%) — importance of days with updates", min_value=0, max_value=100, value=50, step=5)
st.markdown(f"**Quantity weight (%) = {100 - freq_weight_pct}%**")
weight_freq = freq_weight_pct / 100.0
weight_qty = 1.0 - weight_freq

if metrics_cached.empty and not last7_all.empty:
    metrics_df = last7_all.groupby('jalmitra').agg(
        days_updated = ('reading_date', lambda x: x.nunique()),
        total_water_m3 = ('water_quantity', 'sum'),
        schemes_covered = ('scheme_id', lambda x: x.nunique())
    ).reset_index()
else:
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

# Weighted score using UI slider
metrics_df['score'] = metrics_df['days_norm'] * weight_freq + metrics_df['qty_norm'] * weight_qty

metrics_df = metrics_df.sort_values(by=['score','total_water_m3'], ascending=False).reset_index(drop=True)
metrics_df['Rank'] = metrics_df.index + 1
metrics_df['total_water_m3'] = metrics_df['total_water_m3'].round(2)
metrics_df['score'] = metrics_df['score'].round(3)

top_n = 10
top_table = metrics_df.sort_values(by='score', ascending=False).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

worst_table = metrics_df.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

def style_top(df: pd.DataFrame):
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
    sty = sty.set_table_styles([{'selector':'th','props':[('font-weight','600')]}])
    return sty

def style_worst(df: pd.DataFrame):
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds')
    sty = sty.set_table_styles([{'selector':'th','props':[('font-weight','600')]}])
    return sty

colt, colb = st.columns(2)
with colt:
    st.markdown("### 🟢 Top Performers (Best → Worst)")
    if top_table.empty:
        st.info("No top performers to show.")
    else:
        st.dataframe(style_top(top_table), height=420)
        csv_top = top_table.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Top 10 CSV", csv_top, file_name="jjm_top_10.csv", mime="text/csv")

with colb:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to show.")
    else:
        st.dataframe(style_worst(worst_table), height=420)
        csv_worst = worst_table.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Worst 10 CSV", csv_worst, file_name="jjm_worst_10.csv", mime="text/csv")

st.markdown("---")
with st.expander("ℹ️ How ranking is computed (click to expand)"):
    st.markdown(f"""
    - **Days Updated (last 7d)** — distinct days with at least one reading (0–7).
    - **Total Water (m³)** — cumulative water_quantity for the last 7 days.
    - Normalized:
      - `days_norm = days_updated / 7`
      - `qty_norm = total_water / max_total_water`
    - **Score = {weight_freq:.2f} * days_norm + {weight_qty:.2f} * qty_norm**
    - Adjust the **Frequency weight (%)** slider above to change importance. Quantity weight is complementary.
    """)

st.markdown("---")
st.success(f"Dashboard ready. Demo data generated: {st.session_state.get('demo_generated', False)}. Data stored only for this session.")
