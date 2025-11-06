# jjm_demo_app.py
# JJM Dashboard — clickable Jalmitra rows show 7-day performance chart
# - Session-state demo data with SO = "ROKI RAY"
# - Assamese Jalmitra names, Scheme Name column (Village PWSS)
# - Top/Worst tables include a "View" button per row; clicking shows that Jalmitra's 7-day chart

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
st.set_page_config(page_title="JJM Dashboard — Clickable Rows", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("Tap **View** beside any Jalmitra in the Top/Worst lists to see their 7-day performance chart.")
st.markdown("---")

# ---------------------------
# Helper: ensure required columns exist
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
# Helpers
# ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity", "scheme_name"
    ])
    st.session_state["jalmitras"] = []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False
    st.session_state["generating"] = False
    st.session_state["selected_jalmitra"] = None

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

    # PWSS village names for scheme display
    village_names = [
        "Rampur", "Kahikuchi", "Dalgaon", "Guwahati", "Boko", "Moran", "Tezpur", "Sibsagar", "Jorhat", "Hajo",
        "Tihu", "Kokrajhar", "Nalbari", "Barpeta", "Rangia", "Goalpara", "Dhemaji", "Dibrugarh", "Mariani", "Sonari"
    ]

    # create schemes
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
        scheme_label = random.choice(village_names) + " PWSS"
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
                    "water_quantity": round(random.uniform(40.0, 350.0), 2),
                    "scheme_name": scheme_label
                })
                st.session_state["next_reading_id"] += 1

    st.session_state["readings"] = pd.DataFrame(readings_rows)
    st.session_state["demo_generated"] = True

@st.cache_data
def compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date):
    readings_df = ensure_columns(readings_df.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
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
# UI — Demo data management
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
if role != "Section Officer":
    st.header(f"{role} Dashboard — Placeholder")
    st.stop()

# ---------------------------
# Section Officer Dashboard (main)
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "ROKI RAY"

schemes_df = ensure_columns(st.session_state.get("schemes", pd.DataFrame()).copy(), ["id","scheme_name","functionality","so_name"])
readings_df = ensure_columns(st.session_state.get("readings", pd.DataFrame()).copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
jalmitras_list = st.session_state.get("jalmitras", [])

if schemes_df.empty:
    st.info("No schemes found. Generate demo data first.")
    st.stop()

# ---------------------------
# Overview (date + SO name)
# ---------------------------
st.subheader("📋 Overview")
today = datetime.date.today()
today_label = today.strftime("%A, %d %B %Y").upper()
st.markdown(f"**DATE:** {today_label}  **SECTION OFFICER:** {so_name}")

func_counts = schemes_df['functionality'].value_counts()
today_iso = today.isoformat()

merged_today = readings_df.merge(
    schemes_df[['id','scheme_name','functionality','so_name']],
    left_on='scheme_id', right_on='id', how='left'
) if not readings_df.empty else pd.DataFrame()
merged_today = ensure_columns(merged_today, ['reading_date','functionality','so_name','jalmitra','scheme_name'])

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
# Top & Worst Jalmitras (with clickable View buttons)
# ---------------------------
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7_all, metrics_cached = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

# ensure metrics_df exists
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

    # assign random Scheme Name (village PWSS) per jalmitra row
    village_names = [
        "Rampur", "Kahikuchi", "Dalgaon", "Guwahati", "Boko", "Moran", "Tezpur", "Sibsagar", "Jorhat", "Hajo",
        "Tihu", "Kokrajhar", "Nalbari", "Barpeta", "Rangia", "Goalpara", "Dhemaji", "Dibrugarh", "Mariani", "Sonari"
    ]
    # keep assignment deterministic for session by seeding with session id (optional)
    random.seed(42)
    metrics_df['Scheme Name'] = [random.choice(village_names) + " PWSS" for _ in range(len(metrics_df))]

    top_n = 10
    top_df = metrics_df.head(top_n)[['Rank','jalmitra','Scheme Name','days_updated','total_water_m3','score']].copy()
    top_df.columns = ['Rank','Jalmitra','Scheme Name','Days Updated (last 7d)','Total Water (m³)','Score']

    worst_df = metrics_df.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','Scheme Name','days_updated','total_water_m3','score']].copy()
    worst_df.columns = ['Rank','Jalmitra','Scheme Name','Days Updated (last 7d)','Total Water (m³)','Score']

    # Render Top and Worst with a small "View" button per row
    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        # header
        st.write("")  # spacer
        for i, row in top_df.reset_index(drop=True).iterrows():
            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.2,2.2,1.4,1.4,1.0,0.9])
            c0.write(row['Rank'])
            c1.write(row['Jalmitra'])
            c2.write(row['Scheme Name'])
            c3.write(row['Days Updated (last 7d)'])
            c4.write(f"{row['Total Water (m³)']:,}")
            c5.write(f"{row['Score']:.3f}")
            btn_key = f"view_top_{i}_{row['Jalmitra']}"
            if c6.button("View", key=btn_key):
                st.session_state["selected_jalmitra"] = row['Jalmitra']
        st.download_button("⬇️ Download Top 10 CSV", top_df.to_csv(index=False).encode('utf-8'), "top_10_jalmitras.csv")

    with col_w:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.write("")  # spacer
        for i, row in worst_df.reset_index(drop=True).iterrows():
            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.2,2.2,1.4,1.4,1.0,0.9])
            c0.write(row['Rank'])
            c1.write(row['Jalmitra'])
            c2.write(row['Scheme Name'])
            c3.write(row['Days Updated (last 7d)'])
            c4.write(f"{row['Total Water (m³)']:,}")
            c5.write(f"{row['Score']:.3f}")
            btn_key = f"view_worst_{i}_{row['Jalmitra']}"
            if c6.button("View", key=btn_key):
                st.session_state["selected_jalmitra"] = row['Jalmitra']
        st.download_button("⬇️ Download Worst 10 CSV", worst_df.to_csv(index=False).encode('utf-8'), "worst_10_jalmitras.csv")

# ---------------------------
# If user clicked View, show the 7-day performance chart
# ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")
    st.subheader(f"7-day Performance — {jm}")
    # compute daily totals for this jalmitra from last7_all
    jf = last7_all.copy() if 'last7' in locals() else last7_all
    # if last7_all variable name is different due to scope, use compute again
    if jf is None or jf.empty:
        # recompute defensively
        last7_all2, _ = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)
        jf = last7_all2.copy() if not last7_all2.empty else pd.DataFrame()

    if jf.empty:
        st.info("No data available for this Jalmitra in the last 7 days.")
    else:
        jm_data = jf[jf['jalmitra'] == jm].copy()
        if jm_data.empty:
            st.info("No readings for this Jalmitra in the last 7 days.")
        else:
            daily = jm_data.groupby('reading_date')['water_quantity'].sum().reindex(
                pd.date_range(start=(datetime.date.today()-datetime.timedelta(days=6)), periods=7).astype(str),
                fill_value=0
            ).reset_index()
            daily.columns = ['reading_date','water_quantity']
            fig = px.bar(daily, x='reading_date', y='water_quantity', labels={'reading_date':'Date','water_quantity':'Water (m³)'}, title=f"{jm} — Daily Water Supplied (last 7 days)")
            st.plotly_chart(fig, use_container_width=True, height=380)
            # small stats
            total = daily['water_quantity'].sum()
            days_with_updates = int((daily['water_quantity'] > 0).sum())
            st.markdown(f"**Total (7 days):** {total:.2f} m³  **Days updated:** {days_with_updates}/7")
    # allow closing selection
    if st.button("Close View"):
        st.session_state["selected_jalmitra"] = None

st.markdown("---")
st.subheader("📤 Export Snapshot")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), "schemes_snapshot.csv")
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), "readings_snapshot.csv")
try:
    st.download_button("Download Metrics CSV", metrics_df.to_csv(index=False).encode('utf-8'), "metrics_snapshot.csv")
except Exception:
    st.info("Metrics CSV not available (no data).")

st.success(f"Dashboard ready for SO: {so_name}. Demo data generated: {st.session_state.get('demo_generated', False)} ✅")
