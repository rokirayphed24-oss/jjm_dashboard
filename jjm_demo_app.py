# jjm_demo_app.py
# Unified Streamlit app — clickable pies that filter tables
# - Uses streamlit-plotly-events for Plotly click capture (fallback to buttons if not installed)
# - All previous features preserved: session-state demo data, 50/50 weights, Top/Worst tables, Plotly 7-day chart

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import io
import plotly.express as px
from typing import Tuple
from pathlib import Path

# Try to import plotly click event helper; if unavailable use fallback UI
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="JJM Dashboard — Clickable Pies", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard (Clickable Pies)")
st.markdown("Click a pie slice to filter the corresponding table(s). If your environment lacks `streamlit-plotly-events`, fallback buttons are shown.")
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
    # store current pie selections
    if "selected_functionality_slice" not in st.session_state:
        st.session_state["selected_functionality_slice"] = None  # "Functional" / "Non-Functional" / None
    if "selected_updates_slice" not in st.session_state:
        st.session_state["selected_updates_slice"] = None  # "Updated" / "Absent" / None

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
    st.session_state["selected_functionality_slice"] = None
    st.session_state["selected_updates_slice"] = None

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
# Overview: side-by-side pies (clickable) + optional image
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
updated_set = set(today_updates['jalmitra'].unique()) if not today_updates.empty else set()
total_functional = int(len(schemes_df[schemes_df['functionality'] == 'Functional']))
updated_count = len(updated_set)
absent_count = max(total_functional - updated_count, 0)

# Build the Plotly pie figures
fig_func = px.pie(names=func_counts.index, values=func_counts.values, title="Scheme Functionality",
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"}, hole=0.3)
fig_func.update_traces(textinfo='percent+label')
fig_func.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=240)

df_part = pd.DataFrame({"status": ["Updated","Absent"], "count": [updated_count, absent_count]})
if df_part['count'].sum() == 0:
    df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[0, total_functional if total_functional>0 else 1]})
fig_updates = px.pie(df_part, names='status', values='count', title="Jalmitra Updates (Today)",
                     color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"}, hole=0.3)
fig_updates.update_traces(textinfo='percent+label')
fig_updates.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=240)

# Display pies side-by-side and capture clicks (if available)
col_l, col_r = st.columns([1,1])
with col_l:
    st.markdown("#### Scheme Functionality")
    st.plotly_chart(fig_func, use_container_width=True)
    # capture click via streamlit-plotly-events if available
    if PLOTLY_EVENTS_AVAILABLE:
        clicks = plotly_events(fig_func, click_event=True, hover_event=False)
        if clicks:
            # clicks is a list; extract the label clicked (for pie it's 'label' in point)
            label = clicks[0].get('label') or clicks[0].get('x') or clicks[0].get('name')
            # Acceptable labels: 'Functional' or 'Non-Functional'
            if label in ["Functional", "Non-Functional"]:
                st.session_state["selected_functionality_slice"] = label
    else:
        st.info("Tip: install 'streamlit-plotly-events' to enable click-on-pie behaviour. Fallback buttons shown below.")
        if st.button("Show Functional Schemes"):
            st.session_state["selected_functionality_slice"] = "Functional"
        if st.button("Show Non-Functional Schemes"):
            st.session_state["selected_functionality_slice"] = "Non-Functional"

with col_r:
    st.markdown("#### Jalmitra Updates (Today)")
    st.plotly_chart(fig_updates, use_container_width=True)
    if PLOTLY_EVENTS_AVAILABLE:
        clicks2 = plotly_events(fig_updates, click_event=True, hover_event=False)
        if clicks2:
            label2 = clicks2[0].get('label') or clicks2[0].get('x') or clicks2[0].get('name')
            if label2 in ["Updated", "Absent"]:
                st.session_state["selected_updates_slice"] = label2
    else:
        if st.button("Show Updated Jalmitras"):
            st.session_state["selected_updates_slice"] = "Updated"
        if st.button("Show Absent Jalmitras"):
            st.session_state["selected_updates_slice"] = "Absent"

# Small clear selection buttons
col_clear1, col_clear2 = st.columns([1,1])
with col_clear1:
    if st.button("Clear Functionality Filter"):
        st.session_state["selected_functionality_slice"] = None
with col_clear2:
    if st.button("Clear Updates Filter"):
        st.session_state["selected_updates_slice"] = None

st.markdown("---")

# ---------------------------
# Schemes table (responds to functionality pie selection)
# ---------------------------
st.subheader("All Schemes under SO")
if st.session_state["selected_functionality_slice"]:
    sel = st.session_state["selected_functionality_slice"]
    st.markdown(f"**Filtered:** {sel}")
    st.dataframe(schemes_df[schemes_df['functionality'] == sel], height=220)
else:
    st.dataframe(schemes_df, height=220)

st.subheader("Functional Schemes under SO")
functional_schemes = schemes_df[schemes_df['functionality'] == "Functional"]
if functional_schemes.empty:
    st.info("No functional schemes found under this SO.")
else:
    # If functionality filter present but not 'Functional', show empty
    if st.session_state["selected_functionality_slice"] and st.session_state["selected_functionality_slice"] != "Functional":
        st.info("Functionality filter applied (not 'Functional') — no functional schemes to show.")
    else:
        st.dataframe(functional_schemes, height=220)

# ---------------------------
# Today's readings and participation (details adjust with updates pie selection)
# ---------------------------
st.markdown("---")
st.subheader("BFM Readings by Jalmitras Today (Functional schemes)")

readings_today = today_updates[['scheme_name','jalmitra','reading','reading_time','water_quantity']] if not today_updates.empty else pd.DataFrame(columns=['scheme_name','jalmitra','reading','reading_time','water_quantity'])
st.write(f"Total readings recorded today: **{len(readings_today)}**")
if not readings_today.empty:
    st.dataframe(readings_today, height=220)
else:
    st.info("No readings recorded today for functional schemes.")

# Show small details table for Updated or Absent based on selection
st.markdown("---")
st.subheader("Jalmitra Details (based on Updates pie selection)")
selected_updates = st.session_state["selected_updates_slice"]
if selected_updates == "Updated":
    # show distinct updated jalmitras with their total water today
    if not today_updates.empty:
        upd_summary = today_updates.groupby('jalmitra')['water_quantity'].sum().reset_index().rename(columns={'water_quantity':'Water Today (m³)'})
        st.dataframe(upd_summary, height=220)
    else:
        st.info("No updated jalmitras today.")
elif selected_updates == "Absent":
    # compute absent jalmitras among functional schemes
    # assume jalmitra mapping per scheme — find jalmitras mapped to functional schemes and not in updated_set
    jalmitra_mapped = set(functional_schemes.reset_index().index)  # fallback mapping not used; instead map by deterministic JM list presence
    # better approach: jalmitras_list contains JM-1..JM-N; absent defined as those in jalmitras_list not in updated_set
    absent_list = [jm for jm in jalmitras_list if jm not in updated_set]
    if absent_list:
        df_abs = pd.DataFrame({'Jalmitra': absent_list})
        st.dataframe(df_abs, height=220)
    else:
        st.info("No absent jalmitras today.")
else:
    st.info("Click the Updates pie (or use fallback buttons) to view Updated or Absent Jalmitra details here.")

# ---------------------------
# Rankings: Top & Worst (filtered by updates selection if set)
# ---------------------------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")

start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
end_date = today
last7_all, metrics_cached = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all.empty:
    st.info("No readings in the last 7 days for functional schemes. Generate demo data to see rankings.")
else:
    # prepare metrics and ensure all jalmitras present
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

    # apply updates selection filter (if any)
    selected_updates = st.session_state["selected_updates_slice"]
    if selected_updates == "Updated":
        # keep only those who updated today
        metrics_df = metrics_df[metrics_df['jalmitra'].isin(updated_set)].copy()
    elif selected_updates == "Absent":
        metrics_df = metrics_df[~metrics_df['jalmitra'].isin(updated_set)].copy()
    # else show all

    if metrics_df.empty:
        st.info("No Jalmitras match the selected filter.")
    else:
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
            sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
            return sty

        def style_worst(df: pd.DataFrame):
            sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
            sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds_r')
            sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
            return sty

        col_t, col_w = st.columns([1,1])
        with col_t:
            st.markdown("### 🟢 Top 10 Performing Jalmitras")
            st.dataframe(style_top(top_table), height=420)
            st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode('utf-8'), file_name="jjm_top_10.csv", mime="text/csv")
        with col_w:
            st.markdown("### 🔴 Worst 10 Performing Jalmitras")
            st.dataframe(style_worst(worst_table), height=420)
            st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode('utf-8'), file_name="jjm_worst_10.csv", mime="text/csv")

# ---------------------------
# 7-day line chart
# ---------------------------
st.markdown("---")
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

# ---------------------------
# Snapshot exports & info
# ---------------------------
st.markdown("---")
st.subheader("Export Snapshot")
st.markdown("Download current Schemes, Readings, and Metrics as CSVs.")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), file_name='schemes_snapshot.csv', mime='text/csv')
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), file_name='readings_snapshot.csv', mime='text/csv')
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

# Quick note about dependency for native clickable pies
if not PLOTLY_EVENTS_AVAILABLE:
    st.warning("For native pie-click behaviour install 'streamlit-plotly-events' and add it to requirements.txt (package name: streamlit-plotly-events).")

st.success(f"Dashboard ready. Demo data generated: {st.session_state.get('demo_generated', False)}. Data stored only for this session.")
