# jjm_demo_app.py
# Jal Jeevan Mission — Full dashboard (robust, defensive, clickable pies)
# - All features included (see header notes in conversation)
# - Uses SQLite safely; falls back to session-state if DB not usable
# - Clickable pies via streamlit-plotly-events when available (fallback buttons otherwise)
# - Ranking: fixed 50% days-updated + 50% total-water (as requested)
# - Export CSVs, interactive Plotly chart, Top/Worst tables with color gradients
# - Defensive error handling and helpful on-screen debug info

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import io
import os
import logging
from pathlib import Path
from typing import Tuple

# plotting
import plotly.express as px
import matplotlib.pyplot as plt

# optional dependency for clickable plotly events
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

# sqlalchemy for sqlite
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError, SQLAlchemyError

# ---------------------------
# Logging
# ---------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jjm_dashboard")

# ---------------------------
# Config / DB file path
# ---------------------------
st.set_page_config(page_title="JJM Unified Dashboard — Full", layout="wide")
DB_FILENAME = "jjm_demo.sqlite"
DB_PATH = Path(DB_FILENAME)

# Use an app folder for images if present
IMAGE_REL_PATH = "assaa.png"  # include in repo if you want it shown

# ---------------------------
# Helpers: DB engine / session fallback
# ---------------------------
def get_engine():
    """
    Try to create/connect to a SQLite engine in the app folder.
    If that fails (permission), return None and we will use session_state fallback.
    """
    try:
        engine = create_engine(f"sqlite:///{DB_FILENAME}", connect_args={"check_same_thread": False})
        # quick test: create tables if not exist
        with engine.begin() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_name TEXT,
                functionality TEXT,
                so_name TEXT
            )
            """))
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bfm_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_id INTEGER,
                jalmitra TEXT,
                reading INTEGER,
                reading_date TEXT,
                reading_time TEXT,
                water_quantity REAL
            )
            """))
        return engine
    except (OperationalError, SQLAlchemyError, Exception) as e:
        logger.warning("SQLite engine not available or not writable: %s", e)
        return None

# ---------------------------
# Session-state fallback init (only used if DB not available)
# ---------------------------
def init_session_state():
    if "schemes_df" not in st.session_state:
        st.session_state["schemes_df"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name"])
    if "readings_df" not in st.session_state:
        st.session_state["readings_df"] = pd.DataFrame(columns=[
            "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity"
        ])
    if "jalmitras" not in st.session_state:
        st.session_state["jalmitras"] = []
    if "next_scheme_id" not in st.session_state:
        st.session_state["next_scheme_id"] = 1
    if "next_reading_id" not in st.session_state:
        st.session_state["next_reading_id"] = 1
    if "demo_generated" not in st.session_state:
        st.session_state["demo_generated"] = False
    if "selected_functionality_slice" not in st.session_state:
        st.session_state["selected_functionality_slice"] = None
    if "selected_updates_slice" not in st.session_state:
        st.session_state["selected_updates_slice"] = None

# ---------------------------
# DB read/write helpers (use engine if exists, else session-state)
# ---------------------------
engine = get_engine()
use_db = engine is not None

if not use_db:
    init_session_state()
    st.warning("SQLite DB not available or not writable — running in session-only mode. Data will be lost on refresh.")
else:
    st.info("Using SQLite DB file: %s" % DB_FILENAME)

def read_table_sql(table_name: str):
    if use_db:
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)
            return df
        except Exception as e:
            logger.error("Error reading table %s: %s", table_name, e)
            return pd.DataFrame()
    else:
        key = "schemes_df" if table_name == "schemes" else "readings_df"
        return st.session_state.get(key, pd.DataFrame())

def write_schemes_df(df: pd.DataFrame):
    if use_db:
        # replace table
        try:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM schemes"))
                # bulk insert
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO schemes (id, scheme_name, functionality, so_name)
                        VALUES (:id, :scheme_name, :functionality, :so_name)
                    """), {"id": int(row["id"]), "scheme_name": row["scheme_name"], "functionality": row["functionality"], "so_name": row["so_name"]})
        except Exception as e:
            logger.error("Error writing schemes: %s", e)
            # fallback to session state
            st.session_state["schemes_df"] = df.copy()
    else:
        st.session_state["schemes_df"] = df.copy()

def write_readings_df(df: pd.DataFrame):
    if use_db:
        try:
            with engine.begin() as conn:
                conn.execute(text("DELETE FROM bfm_readings"))
                for _, row in df.iterrows():
                    conn.execute(text("""
                        INSERT INTO bfm_readings (id, scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                        VALUES (:id, :scheme_id, :jalmitra, :reading, :reading_date, :reading_time, :water_quantity)
                    """), {
                        "id": int(row["id"]),
                        "scheme_id": int(row["scheme_id"]),
                        "jalmitra": row["jalmitra"],
                        "reading": int(row["reading"]),
                        "reading_date": row["reading_date"],
                        "reading_time": row["reading_time"],
                        "water_quantity": float(row["water_quantity"])
                    })
        except Exception as e:
            logger.error("Error writing readings: %s", e)
            st.session_state["readings_df"] = df.copy()
    else:
        st.session_state["readings_df"] = df.copy()

def append_scheme_row(scheme_row: dict):
    if use_db:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:scheme_name, :functionality, :so_name)
                """), scheme_row)
        except Exception as e:
            logger.error("Error inserting scheme row: %s", e)
            # fallback
            df = st.session_state["schemes_df"]
            df = pd.concat([df, pd.DataFrame([scheme_row])], ignore_index=True)
            st.session_state["schemes_df"] = df
    else:
        df = st.session_state["schemes_df"]
        df = pd.concat([df, pd.DataFrame([scheme_row])], ignore_index=True)
        st.session_state["schemes_df"] = df

def append_reading_row(reading_row: dict):
    if use_db:
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO bfm_readings (scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                    VALUES (:scheme_id, :jalmitra, :reading, :reading_date, :reading_time, :water_quantity)
                """), reading_row)
        except Exception as e:
            logger.error("Error inserting reading row: %s", e)
            df = st.session_state["readings_df"]
            df = pd.concat([df, pd.DataFrame([reading_row])], ignore_index=True)
            st.session_state["readings_df"] = df
    else:
        df = st.session_state["readings_df"]
        df = pd.concat([df, pd.DataFrame([reading_row])], ignore_index=True)
        st.session_state["readings_df"] = df

# ---------------------------
# Utility: create demo data (writes to DB or session-state)
# ---------------------------
def generate_demo_into_store(total_schemes=20, so_name="SO-Guwahati"):
    """
    Generate 20 demo schemes and up to 7 days readings for functional schemes.
    Uses deterministic Jalmitra names JM-1..JM-N mapped to schemes by index.
    """
    today = datetime.date.today()
    FIXED_UPDATE_PROB = 0.85
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    # Schemes dataframe
    schemes = []
    jalmitras = [f"JM-{i+1}" for i in range(total_schemes)]
    next_scheme_id = 1
    # if using DB and existing rows, calculate next id
    if use_db:
        try:
            existing = read_table_sql("schemes")
            if not existing.empty:
                next_scheme_id = int(existing['id'].max()) + 1
        except Exception:
            next_scheme_id = 1
    else:
        if "next_scheme_id" in st.session_state:
            next_scheme_id = st.session_state["next_scheme_id"]

    for i in range(total_schemes):
        scheme_name = f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}"
        functionality = random.choice(["Functional", "Non-Functional"])
        schemes.append({
            "id": next_scheme_id,
            "scheme_name": scheme_name,
            "functionality": functionality,
            "so_name": so_name
        })
        next_scheme_id += 1

    schemes_df = pd.DataFrame(schemes)

    # Save schemes to store (replace)
    if use_db:
        write_schemes_df(schemes_df)
    else:
        st.session_state["schemes_df"] = schemes_df
        st.session_state["jalmitras"] = jalmitras
        st.session_state["next_scheme_id"] = next_scheme_id

    # readings
    readings = []
    next_reading_id = 1
    if use_db:
        try:
            existing_r = read_table_sql("bfm_readings")
            if not existing_r.empty:
                next_reading_id = int(existing_r['id'].max()) + 1
        except Exception:
            next_reading_id = 1
    else:
        if "next_reading_id" in st.session_state:
            next_reading_id = st.session_state["next_reading_id"]

    for idx, row in schemes_df.reset_index().iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_id = int(row["id"])
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                readings.append({
                    "id": next_reading_id,
                    "scheme_id": scheme_id,
                    "jalmitra": jalmitra,
                    "reading": int(random.choice(reading_samples)),
                    "reading_date": date,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity": round(random.uniform(40.0, 350.0), 2)
                })
                next_reading_id += 1

    readings_df = pd.DataFrame(readings)
    if use_db:
        write_readings_df(readings_df)
    else:
        st.session_state["readings_df"] = readings_df
        st.session_state["next_reading_id"] = next_reading_id

    # mark demo_generated
    if use_db:
        # store a tiny marker in a table? we'll just set session flag
        st.session_state["demo_generated"] = True
    else:
        st.session_state["demo_generated"] = True

# ---------------------------
# Utility: read merged tables and metrics (safe)
# ---------------------------
@st.cache_data
def get_schemes_readings(so_name="SO-Guwahati"):
    """
    Return schemes_df and readings_df merged (safe), using DB or session fallback.
    """
    if use_db:
        try:
            with engine.connect() as conn:
                schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name = :so"), conn, params={"so": so_name})
                readings = pd.read_sql(text("SELECT * FROM bfm_readings"), conn)
            return schemes, readings
        except Exception as e:
            logger.error("Error fetching schemes/readings from DB: %s", e)
            # fallback to session state if available
            init_session_state()
            return st.session_state.get("schemes_df", pd.DataFrame()), st.session_state.get("readings_df", pd.DataFrame())
    else:
        init_session_state()
        return st.session_state.get("schemes_df", pd.DataFrame()), st.session_state.get("readings_df", pd.DataFrame())

@st.cache_data
def compute_metrics_and_pivot(readings_df: pd.DataFrame, schemes_df: pd.DataFrame, so_name: str, start_date: str, end_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute merged last7_all and per-jalmitra metrics.
    """
    if readings_df is None or schemes_df is None or schemes_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    try:
        merged = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    except Exception:
        # Defensive: ensure expected columns exist
        merged = readings_df.copy()
        merged['scheme_name'] = merged.get('scheme_name', '')
        merged['functionality'] = merged.get('functionality', '')
        merged['so_name'] = merged.get('so_name', '')

    mask = (merged['functionality'] == 'Functional') & (merged['so_name'] == so_name) & (merged['reading_date'] >= start_date) & (merged['reading_date'] <= end_date)
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
# UI: Header
# ---------------------------
st.title("Jal Jeevan Mission — Landing Dashboard")
st.markdown("---")

# Optional logo/image
if Path(IMAGE_REL_PATH).exists():
    st.image(IMAGE_REL_PATH, width=180)

# ---------------------------
# Demo Data Management (UI)
# ---------------------------
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns(2)
with col1:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=200, value=20, step=1)
    if st.button("Generate Demo Data"):
        try:
            generate_demo_into_store(total_schemes)
            st.success("✅ Demo data generated.")
        except Exception as e:
            st.error("Error generating demo data: %s" % e)
            logger.exception(e)

with col2:
    if st.button("Remove Demo Data"):
        # remove DB tables or clear session
        if use_db:
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM schemes"))
                    conn.execute(text("DELETE FROM bfm_readings"))
                st.success("🗑️ Demo data removed from DB.")
            except Exception as e:
                st.error("Error clearing DB: %s" % e)
                logger.exception(e)
        else:
            init_session_state()
            st.success("🗑️ Demo data removed from session.")

st.markdown("---")

# ---------------------------
# Role selection
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("Currently the Section Officer (SO) view is the main active view. AEE / EE are placeholders.")
    if st.button("Continue to SO view anyway"):
        pass
    else:
        st.stop()

# ---------------------------
# Main Dashboard (SO)
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "SO-Guwahati"

# fetch data
schemes_df, readings_df = get_schemes_readings(so_name=so_name)

if schemes_df is None or schemes_df.empty:
    st.info("No schemes found. Use 'Generate Demo Data' above to populate the dashboard.")
    st.stop()

# ---------- Overview: two pies side-by-side (clickable) ----------
st.subheader("📊 Overview")

# Pie 1: Scheme functionality
func_counts = schemes_df['functionality'].value_counts()
fig_func = px.pie(names=func_counts.index, values=func_counts.values, title="Scheme Functionality",
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"}, hole=0.3)
fig_func.update_traces(textinfo='percent+label')
fig_func.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=260)

# Pie 2: Jalmitra updates today (functional schemes only)
today = datetime.date.today().isoformat()
merged_today = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left') if not readings_df.empty else pd.DataFrame()
today_updates = merged_today[
    (merged_today['reading_date'] == today) &
    (merged_today['functionality'] == 'Functional') &
    (merged_today['so_name'] == so_name)
] if not merged_today.empty else pd.DataFrame()
updated_set = set(today_updates['jalmitra'].unique()) if not today_updates.empty else set()
total_functional = int(len(schemes_df[schemes_df['functionality'] == 'Functional']))
updated_count = len(updated_set)
absent_count = max(total_functional - updated_count, 0)
df_updates = pd.DataFrame({"status": ["Updated", "Absent"], "count": [updated_count, absent_count]})
if df_updates['count'].sum() == 0:
    df_updates = pd.DataFrame({"status":["Updated","Absent"], "count":[0, max(total_functional,1)]})
fig_updates = px.pie(df_updates, names='status', values='count', title="Jalmitra Updates (Today)",
                     color='status', color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"}, hole=0.3)
fig_updates.update_traces(textinfo='percent+label')
fig_updates.update_layout(margin=dict(l=10,r=10,t=20,b=10), height=260)

# display pies side-by-side
col_left, col_right = st.columns([1,1])
with col_left:
    st.markdown("#### Scheme Functionality")
    st.plotly_chart(fig_func, use_container_width=True)
    # clickable capture
    if PLOTLY_EVENTS_AVAILABLE:
        clicks = plotly_events(fig_func, click_event=True, hover_event=False, key="func_pie")
        if clicks:
            lbl = clicks[0].get("label") or clicks[0].get("name")
            if lbl in ["Functional", "Non-Functional"]:
                st.session_state["selected_functionality_slice"] = lbl
    else:
        st.info("Tip: install streamlit-plotly-events for native clicks. Fallback buttons available.")
        if st.button("Filter: Functional"):
            st.session_state["selected_functionality_slice"] = "Functional"
        if st.button("Filter: Non-Functional"):
            st.session_state["selected_functionality_slice"] = "Non-Functional"

with col_right:
    st.markdown("#### Jalmitra Updates (Today)")
    st.plotly_chart(fig_updates, use_container_width=True)
    if PLOTLY_EVENTS_AVAILABLE:
        clicks2 = plotly_events(fig_updates, click_event=True, hover_event=False, key="upd_pie")
        if clicks2:
            lbl2 = clicks2[0].get("label") or clicks2[0].get("name")
            if lbl2 in ["Updated", "Absent"]:
                st.session_state["selected_updates_slice"] = lbl2
    else:
        if st.button("Show Updated Jalmitras"):
            st.session_state["selected_updates_slice"] = "Updated"
        if st.button("Show Absent Jalmitras"):
            st.session_state["selected_updates_slice"] = "Absent"

# small clear selection row
c1, c2 = st.columns([1,1])
with c1:
    if st.button("Clear Functionality Filter"):
        st.session_state["selected_functionality_slice"] = None
with c2:
    if st.button("Clear Updates Filter"):
        st.session_state["selected_updates_slice"] = None

st.markdown("---")

# ---------- Schemes table (respects functionality filter) ----------
st.subheader("All Schemes under SO")
if st.session_state.get("selected_functionality_slice"):
    filt = st.session_state["selected_functionality_slice"]
    st.markdown(f"**Filtered by functionality → {filt}**")
    st.dataframe(schemes_df[schemes_df['functionality'] == filt], height=220)
else:
    st.dataframe(schemes_df, height=220)

st.subheader("Functional Schemes under SO")
functional_schemes = schemes_df[schemes_df['functionality'] == "Functional"]
if functional_schemes.empty:
    st.info("No functional schemes found under this SO.")
else:
    st.dataframe(functional_schemes, height=220)

# ---------- Today's readings (functional only) ----------
st.markdown("---")
st.subheader("BFM Readings by Jalmitras Today")
if today_updates.empty:
    st.info("No readings for today for functional schemes.")
else:
    st.dataframe(today_updates[['scheme_name','jalmitra','reading','reading_time','water_quantity']], height=220)

# simple 3-column water table for today
st.markdown("---")
st.subheader("💧 Water Quantity Supplied (m³) per Jalmitra per Scheme (Today)")
if not today_updates.empty:
    table_simple = today_updates[['jalmitra','scheme_name','water_quantity']].copy()
    table_simple.columns = ['Jalmitra','Scheme','Water Quantity (m³)']
    st.dataframe(table_simple, height=220)
    st.download_button("⬇️ Download Today's Water Table (CSV)", table_simple.to_csv(index=False).encode('utf-8'), file_name="water_today.csv", mime="text/csv")
else:
    st.info("No water quantity measurements for today.")

# ---------- Rankings (last 7 days): top and worst with 50/50 weighting ----------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance Rankings (Last 7 Days)")

start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
end_date = today
last7_all, metrics = compute_metrics_and_pivot(readings_df, schemes_df, so_name, start_date, end_date)

if last7_all.empty:
    st.info("No readings in the last 7 days for functional schemes. Generate demo data to see rankings.")
else:
    # ensure all pendings jalmitras exist in metrics
    jalmitras_list = st.session_state.get("jalmitras", [f"JM-{i+1}" for i in range(len(schemes_df))])
    for jm in jalmitras_list:
        if jm not in metrics['jalmitra'].values:
            metrics = pd.concat([metrics, pd.DataFrame([{'jalmitra': jm, 'days_updated': 0, 'total_water_m3': 0.0, 'schemes_covered': 0}])], ignore_index=True)

    metrics['days_norm'] = metrics['days_updated'] / 7.0
    max_qty = metrics['total_water_m3'].max() if not metrics['total_water_m3'].empty else 0.0
    metrics['qty_norm'] = metrics['total_water_m3'] / max_qty if max_qty > 0 else 0.0
    # fixed 50/50 weights
    weight_freq = 0.5
    weight_qty = 0.5
    metrics['score'] = metrics['days_norm'] * weight_freq + metrics['qty_norm'] * weight_qty

    # optionally filter by Updates pie selection (Updated / Absent)
    sel_updates = st.session_state.get("selected_updates_slice")
    if sel_updates == "Updated":
        metrics = metrics[metrics['jalmitra'].isin(updated_set)].copy()
    elif sel_updates == "Absent":
        metrics = metrics[~metrics['jalmitra'].isin(updated_set)].copy()

    if metrics.empty:
        st.info("No Jalmitras match the applied filter.")
    else:
        metrics = metrics.sort_values(by=['score','total_water_m3'], ascending=False).reset_index(drop=True)
        metrics['Rank'] = metrics.index + 1
        metrics['total_water_m3'] = metrics['total_water_m3'].round(2)
        metrics['score'] = metrics['score'].round(3)

        top_table = metrics.sort_values(by='score', ascending=False).head(10)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
        top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

        worst_table = metrics.sort_values(by='score', ascending=True).head(10)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
        worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

        # styling helpers: top = green gradient, worst = dark->light red
        def style_top(df):
            sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
            sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
            return sty

        def style_worst(df):
            # reversed Reds so smaller numeric values -> darker red
            sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
            sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds_r')
            return sty

        colt, colw = st.columns([1,1])
        with colt:
            st.markdown("### 🟢 Top 10 Performing Jalmitras")
            st.dataframe(style_top(top_table), height=420)
            st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode('utf-8'), file_name="top_10.csv", mime="text/csv")
        with colw:
            st.markdown("### 🔴 Worst 10 Performing Jalmitras")
            st.dataframe(style_worst(worst_table), height=420)
            st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode('utf-8'), file_name="worst_10.csv", mime="text/csv")

# ---------- Last 7 days interactive Plotly line chart ----------
st.markdown("---")
st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")

if last7_all.empty:
    st.info("No 7-day data available.")
else:
    last_week_qty = last7_all.groupby(['reading_date','scheme_name'])['water_quantity'].sum().reset_index()
    pivot_chart = last_week_qty.pivot(index='reading_date', columns='scheme_name', values='water_quantity').fillna(0)
    pivot_chart = pivot_chart.sort_index()

    st.markdown("**Chart options**")
    cc1, cc2 = st.columns([2,1])
    with cc1:
        show_total = st.checkbox("Also show total (sum of all functional schemes)", value=True)
        top_k = st.selectbox("Show top N schemes by total water (last 7 days) or 'All'", options=["All","Top 5","Top 10","Top 15"], index=1)
    with cc2:
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

    fig = px.line(plot_df, x='reading_date', y='water_quantity', color='scheme_name', markers=True,
                  labels={'reading_date': 'Date', 'water_quantity': 'Water (m³)', 'scheme_name': 'Scheme'},
                  title="Water Supplied (m³) — last 7 days")
    fig.update_layout(legend_title_text='Scheme / Total')
    if date_order == "Descending":
        fig.update_xaxes(categoryorder='array', categoryarray=sorted(plot_df['reading_date'].unique(), reverse=True))
    st.plotly_chart(fig, use_container_width=True, height=420)

# ---------- Export snapshot & notes ----------
st.markdown("---")
st.subheader("Export Snapshot")
st.download_button("Download Schemes CSV", schemes_df.to_csv(index=False).encode('utf-8'), file_name='schemes_snapshot.csv', mime='text/csv')
st.download_button("Download Readings CSV", readings_df.to_csv(index=False).encode('utf-8'), file_name='readings_snapshot.csv', mime='text/csv')
try:
    st.download_button("Download Metrics CSV", metrics.to_csv(index=False).encode('utf-8'), file_name='metrics_snapshot.csv', mime='text/csv')
except Exception:
    st.info("Metrics CSV not available (no data).")

st.markdown("---")
with st.expander("ℹ️ How ranking is computed"):
    st.markdown("""
    - Days Updated (last 7d) = distinct days a Jalmitra submitted >=1 reading (0-7).
    - Total Water (m³) = sum of water_quantity over last 7 days.
    - Normalization:
        - days_norm = days_updated / 7
        - qty_norm = total_water / max_total_water
    - Score = 0.50 * days_norm + 0.50 * qty_norm (fixed)
    - Top table sorts by score descending; Worst table sorts by score ascending.
    """)

# ------------- Deployment note about clickable pies -------------
if not PLOTLY_EVENTS_AVAILABLE:
    st.warning("For native clickable pie behaviour install 'streamlit-plotly-events' and add it to requirements.txt. Fallback buttons are available in the UI.")

st.success("Dashboard ready. Data stored in SQLite if writable; otherwise stored for session only.")
