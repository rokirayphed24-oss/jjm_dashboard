# jjm_demo_app.py
# Full JJM dashboard — Auto-generate demo data if none exists and show debug info
# - Uses SQLite if writable, else session-state fallback
# - Auto-populates demo data on first load so UI is never empty
# - Clickable pies via streamlit-plotly-events if installed (fallback buttons)
# - Ranking fixed to 50% days-updated + 50% quantity
# - Top/Worst tables with green and dark->light red gradients
# - Explicit Debug panel shows storage mode, counts, sample rows

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from pathlib import Path
from typing import Tuple
import plotly.express as px
import logging

# optional click helper
try:
    from streamlit_plotly_events import plotly_events
    PLOTLY_EVENTS_AVAILABLE = True
except Exception:
    PLOTLY_EVENTS_AVAILABLE = False

# sqlalchemy for SQLite
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("jjm_demo_app")

st.set_page_config(page_title="JJM Dashboard — Auto Demo", layout="wide")

DB_FILE = "jjm_demo.sqlite"
IMAGE_REL_PATH = "assaa.png"

SCHEMES_COLS = ["id", "scheme_name", "functionality", "so_name"]
READINGS_COLS = ["id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"]

# -------------------------
# Utilities
# -------------------------
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

def try_get_engine(db_file=DB_FILE):
    try:
        engine = create_engine(f"sqlite:///{db_file}", connect_args={"check_same_thread": False})
        # ensure tables
        with engine.begin() as conn:
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS schemes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_name TEXT,
                functionality TEXT,
                so_name TEXT
            )"""))
            conn.execute(text("""
            CREATE TABLE IF NOT EXISTS bfm_readings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scheme_id INTEGER,
                jalmitra TEXT,
                reading INTEGER,
                reading_date TEXT,
                reading_time TEXT,
                water_quantity REAL
            )"""))
        return engine
    except Exception as e:
        logger.warning("SQLite engine unavailable: %s", e)
        return None

engine = try_get_engine()
USE_DB = engine is not None

# session fallback storage keys
def init_session():
    if "schemes_df" not in st.session_state:
        st.session_state["schemes_df"] = pd.DataFrame(columns=SCHEMES_COLS)
    if "readings_df" not in st.session_state:
        st.session_state["readings_df"] = pd.DataFrame(columns=READINGS_COLS)
    if "jalmitras" not in st.session_state:
        st.session_state["jalmitras"] = []
    if "demo_generated" not in st.session_state:
        st.session_state["demo_generated"] = False
    if "selected_functionality_slice" not in st.session_state:
        st.session_state["selected_functionality_slice"] = None
    if "selected_updates_slice" not in st.session_state:
        st.session_state["selected_updates_slice"] = None

if not USE_DB:
    init_session()

def read_table(table_name: str) -> pd.DataFrame:
    if USE_DB:
        try:
            with engine.connect() as conn:
                df = pd.read_sql(text(f"SELECT * FROM {table_name}"), conn)
            if table_name == "schemes":
                df = ensure_columns(df, SCHEMES_COLS)
            else:
                df = ensure_columns(df, READINGS_COLS)
            return df
        except Exception as e:
            logger.error("DB read error: %s", e)
    # fallback
    init_session()
    return st.session_state.get("schemes_df" if table_name=="schemes" else "readings_df", pd.DataFrame())

def write_replace(table_name: str, df: pd.DataFrame):
    df = df.copy()
    if table_name == "schemes":
        df = ensure_columns(df, SCHEMES_COLS)
    else:
        df = ensure_columns(df, READINGS_COLS)

    if USE_DB:
        try:
            with engine.begin() as conn:
                conn.execute(text(f"DELETE FROM {table_name}"))
                if not df.empty:
                    if table_name == "schemes":
                        for _, r in df.iterrows():
                            conn.execute(text("""
                                INSERT INTO schemes (id, scheme_name, functionality, so_name)
                                VALUES (:id, :scheme_name, :functionality, :so_name)
                            """), {"id": int(r["id"]) if r["id"]!="" else None, "scheme_name": r["scheme_name"], "functionality": r["functionality"], "so_name": r["so_name"]})
                    else:
                        for _, r in df.iterrows():
                            conn.execute(text("""
                                INSERT INTO bfm_readings (id, scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                                VALUES (:id, :scheme_id, :jalmitra, :reading, :reading_date, :reading_time, :water_quantity)
                            """), {"id": int(r["id"]) if r["id"]!="" else None,
                                  "scheme_id": int(r["scheme_id"]) if r["scheme_id"]!="" else None,
                                  "jalmitra": r["jalmitra"], "reading": int(r["reading"]) if r["reading"]!="" else None,
                                  "reading_date": r["reading_date"], "reading_time": r["reading_time"],
                                  "water_quantity": float(r["water_quantity"]) if r["water_quantity"]!="" else 0.0})
            return
        except Exception as e:
            logger.error("DB write error: %s", e)
    # fallback session
    init_session()
    if table_name == "schemes":
        st.session_state["schemes_df"] = df
    else:
        st.session_state["readings_df"] = df

# -------------------------
# Demo data generator (deterministic mapping)
# -------------------------
def generate_demo_data(total_schemes=20, so_name="SO-Guwahati"):
    today = datetime.date.today()
    FIXED_UPDATE_PROB = 0.85
    reading_samples = [110010,215870,150340,189420,200015,234870]
    schemes = []
    for i in range(total_schemes):
        schemes.append({"id": i+1, "scheme_name": f"Scheme {chr(65 + (i%26))}{'' if i<26 else i//26}", "functionality": random.choice(["Functional","Non-Functional"]), "so_name": so_name})
    schemes_df = pd.DataFrame(schemes)
    schemes_df = ensure_columns(schemes_df, SCHEMES_COLS)

    jalmitras = [f"JM-{i+1}" for i in range(total_schemes)]
    readings = []
    rid = 1
    for idx, row in schemes_df.iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_id = int(row["id"])
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                readings.append({"id": rid, "scheme_id": scheme_id, "jalmitra": jalmitra, "reading": int(random.choice(reading_samples)), "reading_date": date, "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00", "water_quantity": round(random.uniform(40.0,350.0),2)})
                rid += 1
    readings_df = pd.DataFrame(readings)
    readings_df = ensure_columns(readings_df, READINGS_COLS)

    write_replace("schemes", schemes_df)
    write_replace("bfm_readings", readings_df)

    init_session()
    st.session_state["jalmitras"] = jalmitras
    st.session_state["demo_generated"] = True
    st.success("✅ Demo data created (auto).")

# -------------------------
# Compute merged last7 & metrics
# -------------------------
@st.cache_data
def compute_last7_metrics(so_name:str, start_date:str, end_date:str) -> Tuple[pd.DataFrame,pd.DataFrame]:
    schemes_df = read_table("schemes")
    readings_df = read_table("bfm_readings")
    schemes_df = ensure_columns(schemes_df, SCHEMES_COLS)
    readings_df = ensure_columns(readings_df, READINGS_COLS)
    try:
        merged = readings_df.merge(schemes_df[["id","scheme_name","functionality","so_name"]], left_on="scheme_id", right_on="id", how="left")
    except Exception:
        merged = readings_df.copy()
        for c in ["scheme_name","functionality","so_name"]:
            if c not in merged.columns:
                merged[c] = ""
    mask = (merged["functionality"]=="Functional") & (merged["so_name"]==so_name) & (merged["reading_date"]>=start_date) & (merged["reading_date"]<=end_date)
    last7 = merged.loc[mask].copy()
    if last7.empty:
        return last7, pd.DataFrame()
    metrics = last7.groupby("jalmitra").agg(days_updated=("reading_date", lambda x: x.nunique()), total_water_m3=("water_quantity","sum")).reset_index()
    metrics["days_updated"] = metrics["days_updated"].astype(int)
    metrics["total_water_m3"] = metrics["total_water_m3"].astype(float).round(2)
    return last7, metrics

# -------------------------
# UI: header & debug panel
# -------------------------
st.title("Jal Jeevan Mission — Dashboard (Auto Demo)")
st.markdown("---")

# show image if present
if Path(IMAGE_REL_PATH).exists():
    st.image(IMAGE_REL_PATH, width=180)

# debug / status info
col_s, col_m = st.columns([1,2])
with col_s:
    if USE_DB:
        st.success("Storage: SQLite DB (app folder)")
    else:
        st.warning("Storage: SESSION-STATE (ephemeral). Add SQLite write permissions for persistence if needed.")
with col_m:
    st.markdown("**Quick help:** If you see empty messages, click **Generate Demo Data** below or refresh. If using Streamlit Cloud, add `streamlit-plotly-events` to requirements.txt for clickable pies.")

# Immediately auto-generate demo data if store empty (so page isn't blank)
schemes_df = read_table("schemes")
readings_df = read_table("bfm_readings")
schemes_df = ensure_columns(schemes_df, SCHEMES_COLS)
readings_df = ensure_columns(readings_df, READINGS_COLS)

if schemes_df.empty or readings_df.empty:
    st.info("No data detected — auto-generating demo data so dashboard shows examples.")
    generate_demo_data(total_schemes=20)

# Re-read after potential generation
schemes_df = read_table("schemes")
readings_df = read_table("bfm_readings")
schemes_df = ensure_columns(schemes_df, SCHEMES_COLS)
readings_df = ensure_columns(readings_df, READINGS_COLS)

# show sample counts and first rows for clarity (debug)
with st.expander("🔍 Debug: storage & sample data (click to open)", expanded=False):
    st.write("Storage mode:", "SQLite DB" if USE_DB else "Session-state (ephemeral)")
    st.write("Schemes count:", len(schemes_df))
    st.write("Readings count:", len(readings_df))
    st.write("Sample schemes:")
    st.dataframe(schemes_df.head(10))
    st.write("Sample readings:")
    st.dataframe(readings_df.head(10))

st.markdown("---")

# -------------------------
# Demo Data controls
# -------------------------
st.markdown("### 🧪 Demo Data Controls")
c1, c2 = st.columns([2,1])
with c1:
    total_schemes = st.number_input("Total demo schemes to generate", min_value=4, max_value=200, value=20)
    if st.button("Generate Demo Data (manual)"):
        generate_demo_data(total_schemes=int(total_schemes))
with c2:
    if st.button("Clear Demo Data"):
        if USE_DB:
            try:
                with engine.begin() as conn:
                    conn.execute(text("DELETE FROM schemes"))
                    conn.execute(text("DELETE FROM bfm_readings"))
                st.success("DB demo data cleared.")
            except Exception as e:
                st.error("Could not clear DB: %s" % e)
        else:
            init_session()
            st.success("Session demo data cleared.")

st.markdown("---")

# -------------------------
# Role selection (SO primary)
# -------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("AEE/EE are placeholders. Use Section Officer view for full features.")
    if not st.button("Continue to SO view anyway"):
        st.stop()

so_name = "SO-Guwahati"

# -------------------------
# Prepare merged and metrics for last 7 days
# -------------------------
today = datetime.date.today().isoformat()
start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
last7_all, metrics = compute_last7_metrics(so_name, start_date, today)

# scheme functionality pie
func_counts = schemes_df["functionality"].value_counts()
if func_counts.empty:
    func_counts = pd.Series({"Functional":0,"Non-Functional":0})
fig_func = px.pie(names=func_counts.index, values=func_counts.values, hole=0.3, color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
fig_func.update_traces(textinfo='percent+label')
fig_func.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=240)

# updates pie (today) for functional schemes
try:
    merged_today = readings_df.merge(schemes_df[["id","scheme_name","functionality","so_name"]], left_on="scheme_id", right_on="id", how="left") if not readings_df.empty else pd.DataFrame()
except Exception:
    merged_today = readings_df.copy()
    for c in ["scheme_name","functionality","so_name"]:
        if c not in merged_today.columns:
            merged_today[c] = ""

today_updates = merged_today[(merged_today["reading_date"]==today) & (merged_today.get("functionality","")=="Functional") & (merged_today.get("so_name","")==so_name)] if not merged_today.empty else pd.DataFrame()
updated_set = set(today_updates["jalmitra"].unique()) if not today_updates.empty else set()
total_functional = int(len(schemes_df[schemes_df["functionality"]=="Functional"]))
updated_count = len(updated_set)
absent_count = max(total_functional - updated_count, 0)
df_updates = pd.DataFrame({"status":["Updated","Absent"], "count":[updated_count, absent_count]})
if df_updates["count"].sum() == 0:
    df_updates = pd.DataFrame({"status":["Updated","Absent"], "count":[0, max(total_functional,1)]})

fig_updates = px.pie(df_updates, names="status", values="count", hole=0.3, color="status", color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
fig_updates.update_traces(textinfo='percent+label')
fig_updates.update_layout(margin=dict(l=10,r=10,t=10,b=10), height=240)

# show pies side by side & capture clicks (or fallback buttons)
st.subheader("📊 Overview")
colL, colR = st.columns([1,1])
with colL:
    st.markdown("#### Scheme Functionality")
    st.plotly_chart(fig_func, use_container_width=True)
    if PLOTLY_EVENTS_AVAILABLE:
        clicks = plotly_events(fig_func, click_event=True, hover_event=False, key="func")
        if clicks:
            lab = clicks[0].get("label") or clicks[0].get("name")
            if lab in ["Functional","Non-Functional"]:
                st.session_state["selected_functionality_slice"] = lab
    else:
        if st.button("Filter: Functional"):
            st.session_state["selected_functionality_slice"] = "Functional"
        if st.button("Filter: Non-Functional"):
            st.session_state["selected_functionality_slice"] = "Non-Functional"

with colR:
    st.markdown("#### Jalmitra Updates (Today)")
    st.plotly_chart(fig_updates, use_container_width=True)
    if PLOTLY_EVENTS_AVAILABLE:
        clicks2 = plotly_events(fig_updates, click_event=True, hover_event=False, key="upd")
        if clicks2:
            lab2 = clicks2[0].get("label") or clicks2[0].get("name")
            if lab2 in ["Updated","Absent"]:
                st.session_state["selected_updates_slice"] = lab2
    else:
        if st.button("Show Updated"):
            st.session_state["selected_updates_slice"] = "Updated"
        if st.button("Show Absent"):
            st.session_state["selected_updates_slice"] = "Absent"

col_clear1, col_clear2 = st.columns([1,1])
with col_clear1:
    if st.button("Clear Functionality Filter"):
        st.session_state["selected_functionality_slice"] = None
with col_clear2:
    if st.button("Clear Updates Filter"):
        st.session_state["selected_updates_slice"] = None

st.markdown("---")

# Schemes table (respects functionality slice)
st.subheader("All Schemes under SO")
selected_func = st.session_state.get("selected_functionality_slice")
if selected_func:
    st.markdown(f"**Filtered: {selected_func}**")
    st.dataframe(schemes_df[schemes_df["functionality"]==selected_func], height=240)
else:
    st.dataframe(schemes_df, height=240)

st.subheader("Functional Schemes under SO")
st.dataframe(schemes_df[schemes_df["functionality"]=="Functional"], height=220)

st.markdown("---")
st.subheader("BFM Readings by Jalmitras Today (Functional schemes)")
if today_updates.empty:
    st.info("No readings recorded today for functional schemes.")
else:
    st.dataframe(today_updates[["scheme_name","jalmitra","reading","reading_time","water_quantity"]], height=220)

st.markdown("---")
st.subheader("💧 Water Quantity Supplied (m³) per Jalmitra per Scheme (Today)")
if not today_updates.empty:
    simple = today_updates[["jalmitra","scheme_name","water_quantity"]].copy()
    simple.columns = ["Jalmitra","Scheme","Water Quantity (m³)"]
    st.dataframe(simple, height=220)
    st.download_button("Download Today's Water Table CSV", simple.to_csv(index=False).encode("utf-8"), file_name="water_today.csv", mime="text/csv")
else:
    st.info("No water quantity data for today.")

# Rankings (last 7 days) with fixed 50/50 weights
st.markdown("---")
st.subheader("🏅 Jalmitra Rankings (Last 7 Days) — 50% Frequency + 50% Quantity")

if last7_all.empty:
    st.info("No last-7-day readings for functional schemes. Generate demo data.")
else:
    metrics = ensure_columns(metrics, ["jalmitra","days_updated","total_water_m3"])
    # include all jalmitras
    jms = st.session_state.get("jalmitras", [f"JM-{i+1}" for i in range(len(schemes_df))])
    for jm in jms:
        if jm not in metrics["jalmitra"].values:
            metrics = pd.concat([metrics, pd.DataFrame([{"jalmitra":jm,"days_updated":0,"total_water_m3":0.0}])], ignore_index=True)
    metrics["days_norm"] = metrics["days_updated"]/7.0
    max_qty = metrics["total_water_m3"].max() if not metrics["total_water_m3"].empty else 0.0
    metrics["qty_norm"] = metrics["total_water_m3"]/max_qty if max_qty>0 else 0.0
    metrics["score"] = 0.5*metrics["days_norm"] + 0.5*metrics["qty_norm"]

    sel_updates = st.session_state.get("selected_updates_slice")
    if sel_updates == "Updated":
        metrics = metrics[metrics["jalmitra"].isin(updated_set)].copy()
    elif sel_updates == "Absent":
        metrics = metrics[~metrics["jalmitra"].isin(updated_set)].copy()

    if metrics.empty:
        st.info("No Jalmitras match applied filter.")
    else:
        metrics = metrics.sort_values(by=["score","total_water_m3"], ascending=False).reset_index(drop=True)
        metrics["Rank"] = metrics.index+1
        metrics["total_water_m3"] = metrics["total_water_m3"].round(2)
        metrics["score"] = metrics["score"].round(3)

        top = metrics.sort_values(by="score", ascending=False).head(10)[["Rank","jalmitra","days_updated","total_water_m3","score"]].copy()
        top.columns = ["Rank","Jalmitra","Days Updated (last 7d)","Total Water (m³)","Score"]
        worst = metrics.sort_values(by="score", ascending=True).head(10)[["Rank","jalmitra","days_updated","total_water_m3","score"]].copy()
        worst.columns = ["Rank","Jalmitra","Days Updated (last 7d)","Total Water (m³)","Score"]

        def style_g(df):
            s = df.style.format({"Total Water (m³)":"{:,.2f}","Score":"{:.3f}"}).background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Greens")
            return s

        def style_r(df):
            s = df.style.format({"Total Water (m³)":"{:,.2f}","Score":"{:.3f}"}).background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Reds_r")
            return s

        cA, cB = st.columns([1,1])
        with cA:
            st.markdown("### 🟢 Top 10")
            st.dataframe(style_g(top), height=420)
            st.download_button("Download Top 10 CSV", top.to_csv(index=False).encode("utf-8"), file_name="top10.csv", mime="text/csv")
        with cB:
            st.markdown("### 🔴 Worst 10")
            st.dataframe(style_r(worst), height=420)
            st.download_button("Download Worst 10 CSV", worst.to_csv(index=False).encode("utf-8"), file_name="worst10.csv", mime="text/csv")

# 7-day chart
st.markdown("---")
st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")
if last7_all.empty:
    st.info("No 7-day data to chart.")
else:
    lw = last7_all.groupby(["reading_date","scheme_name"])["water_quantity"].sum().reset_index()
    pivot = lw.pivot(index="reading_date", columns="scheme_name", values="water_quantity").fillna(0)
    # simple plotly line
    st.markdown("Chart options:")
    colx1, colx2 = st.columns([2,1])
    with colx1:
        show_total = st.checkbox("Show Total", value=True)
        top_k = st.selectbox("Top N schemes", options=["All","Top 5","Top 10"], index=1)
    with colx2:
        date_order = st.radio("Date order", options=["Ascending","Descending"], index=0)
    scheme_sums = lw.groupby("scheme_name")["water_quantity"].sum().sort_values(ascending=False)
    if top_k == "All":
        sel_schemes = scheme_sums.index.tolist()
    else:
        k = int(top_k.split()[1])
        sel_schemes = scheme_sums.head(k).index.tolist()
    plot_df = lw[lw["scheme_name"].isin(sel_schemes)].copy()
    if show_total:
        td = lw.groupby("reading_date")["water_quantity"].sum().reset_index()
        td["scheme_name"] = "Total (all)"
        plot_df = pd.concat([plot_df, td], ignore_index=True)
    fig = px.line(plot_df, x="reading_date", y="water_quantity", color="scheme_name", markers=True, title="Last 7 days")
    if date_order == "Descending":
        fig.update_xaxes(categoryorder="array", categoryarray=sorted(plot_df["reading_date"].unique(), reverse=True))
    st.plotly_chart(fig, use_container_width=True, height=420)

st.markdown("---")
st.success("If you still see blanks after this, copy the Debug panel info and paste here so I can diagnose logs/permissions precisely.")
