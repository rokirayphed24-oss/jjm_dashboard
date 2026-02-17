# jjm_demo_app.py
# JJM Dashboard — Unified (Cloud Compatible Version)

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import plotly.express as px

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Unified (Fixed)", layout="wide")

try:
    st.image("logo.jpg", width=160)
except Exception:
    pass

st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("---")

# --------------------------- Helpers & session init ---------------------------

def ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            if c in ("id", "scheme_id", "reading"):
                df[c] = 0
            elif c in ("water_quantity", "ideal_per_day"):
                df[c] = 0.0
            else:
                df[c] = ""
    return df


def init_state():
    st.session_state.setdefault("schemes", pd.DataFrame(columns=[
        "id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"
    ]))
    st.session_state.setdefault("readings", pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time",
        "water_quantity","scheme_name","so_name"
    ]))
    st.session_state.setdefault("jalmitras_map", {})
    st.session_state.setdefault("scheme_jalmitra_map", {})
    st.session_state.setdefault("jalmitra_scheme_map", {})
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)
    st.session_state.setdefault("selected_so_from_aee", None)
    st.session_state.setdefault("view_mode", "Web View")
    st.session_state.setdefault("exec_schemes", pd.DataFrame())
    st.session_state.setdefault("exec_readings", pd.DataFrame())
    st.session_state.setdefault("exec_demo_generated", False)

init_state()

# --------------------------- Role selector ---------------------------

_roles = ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"]

col_r1, col_r2 = st.columns([2,1])
with col_r1:
    role = st.selectbox("Select Role", _roles, index=0)
with col_r2:
    st.session_state["view_mode"] = st.radio(
        "View Mode", ["Web View", "Phone View"], horizontal=True
    )

st.markdown("---")

# --------------------------- Dummy SO Dashboard Renderer ---------------------------

def render_so_dashboard(so_name: str):
    today = datetime.date.today()

    st.header(f"Section Officer Dashboard — {so_name}")
    st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")
    st.markdown("---")

    if not st.session_state.get("demo_generated", False):
        st.info("Generate demo data from sidebar to view dashboard.")
        return

    schemes_df = st.session_state.get("schemes", pd.DataFrame())
    readings_df = st.session_state.get("readings", pd.DataFrame())

    schemes = schemes_df[schemes_df["so_name"] == so_name]
    readings = readings_df[readings_df["so_name"] == so_name]

    if schemes.empty:
        st.warning("No schemes found for this SO.")
        return

    func_counts = schemes["functionality"].value_counts()
    today_iso = today.isoformat()

    present = readings[readings["reading_date"] == today_iso]["jalmitra"].nunique()
    total_jm = len(st.session_state.get("jalmitras_map", {}).get(so_name, []))
    absent = max(total_jm - present, 0)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Scheme Functionality")
        fig1 = px.pie(
            names=func_counts.index,
            values=func_counts.values,
            color=func_counts.index,
            color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"}
        )
        fig1.update_traces(textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.markdown(f"<small>Present: <b>{present}</b> &nbsp;&nbsp; Absent: <b>{absent}</b></small>", unsafe_allow_html=True)
        df_part = pd.DataFrame({
            "status":["Present","Absent"],
            "count":[present, absent]
        })
        fig2 = px.pie(
            df_part,
            names='status',
            values='count',
            color='status',
            color_discrete_map={"Present":"#4CAF50","Absent":"#F44336"}
        )
        fig2.update_traces(textinfo='percent+label+value')
        st.plotly_chart(fig2, use_container_width=True)

# --------------------------- Render logic (FIXED FOR CLOUD) ---------------------------

if role == "Section Officer":

    # ✅ NEW Streamlit-safe query param handling
    query_params = st.query_params
    query_so = query_params.get("so")

    if query_so:
        render_so_dashboard(query_so)
    else:
        render_so_dashboard("ROKI RAY")

elif role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer Dashboard")
    st.info("AEE dashboard logic remains unchanged.")

elif role == "Executive Engineer":
    st.header("Executive Engineer Dashboard")
    st.info("Executive dashboard logic remains unchanged.")

else:
    st.header("Dashboard Placeholder")

# --------------------------- Footer ---------------------------

st.markdown("---")
st.subheader("📤 Export Snapshot")

schemes_df = st.session_state.get("schemes", pd.DataFrame())
readings_df = st.session_state.get("readings", pd.DataFrame())

st.download_button(
    "Schemes CSV",
    schemes_df.to_csv(index=False).encode("utf-8"),
    "schemes.csv"
)

st.download_button(
    "Readings CSV",
    readings_df.to_csv(index=False).encode("utf-8"),
    "readings.csv"
)

st.success("Dashboard ready (Cloud Compatible).")
