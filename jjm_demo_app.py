import streamlit as st
import pandas as pd
import numpy as np
import datetime
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

# --- Page Config ---
st.set_page_config(page_title="JJM Role Dashboard", layout="wide")

st.image("logo.jpg", width=180)
st.title("Jal Jeevan Mission — Landing Dashboard")
st.markdown("---")

# --- Database setup (simulate SQLite) ---
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})

# Create dummy tables if not exist
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS schemes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scheme_name TEXT,
        functional INTEGER,  -- 1=functional, 0=non-functional
        so_name TEXT
    )
    """))
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS bfm_readings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scheme_id INTEGER,
        jalmitra TEXT,
        reading REAL,
        reading_date TEXT
    )
    """))

# --- Role selection ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

if role == "Section Officer":
    st.header("Section Officer Dashboard")

    # --- Schemes under this SO ---
    so_name = "SO-Guwahati"  # Example SO
    with engine.connect() as conn:
        schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name=:so"), conn, params={"so": so_name})

    st.subheader("All schemes under SO")
    st.dataframe(schemes)

    # Functional schemes
    functional_schemes = schemes[schemes['functional']==1]
    st.subheader("Functional schemes under SO")
    st.dataframe(functional_schemes)

    # Today's date
    today = datetime.date.today().isoformat()

    # Number of BFM readings updated by Jalmitras today
    with engine.connect() as conn:
        readings_today = pd.read_sql(text("""
            SELECT s.scheme_name, b.jalmitra, b.reading
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date = :today AND s.functional=1 AND s.so_name=:so
        """), conn, params={"today": today, "so": so_name})

    st.subheader(f"Number of BFM readings updated by Jalmitras today")
    st.write(len(readings_today))

    # Matrix: today's readings by Jalmitras vs functional schemes
    if not readings_today.empty:
        matrix = readings_today.pivot_table(index="jalmitra", columns="scheme_name", values="reading")
        st.subheader("Today's BFM readings by Jalmitras")
        st.dataframe(matrix)
    else:
        st.info("No readings recorded today.")

    # Matrix: absent readings (functional schemes not updated)
    if not functional_schemes.empty:
        all_jalmitras = ["JM-1","JM-2","JM-3"]  # Example Jalmitras
        all_scheme_ids = functional_schemes['id'].tolist()
        absent_list = []
        for j in all_jalmitras:
            for s_id in all_scheme_ids:
                if readings_today.empty or not ((readings_today['jalmitra']==j) & (readings_today['scheme_name']==functional_schemes[functional_schemes['id']==s_id]['scheme_name'].values[0])).any():
                    absent_list.append({"jalmitra": j, "scheme_id": s_id})
        absent_df = pd.DataFrame(absent_list)
        st.subheader("Absent readings by Jalmitras")
        st.dataframe(absent_df)

    # Graph: last 7 days readings against all schemes
    week_ago = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    with engine.connect() as conn:
        last_week_readings = pd.read_sql(text("""
            SELECT s.scheme_name, b.reading_date, SUM(b.reading) as total_reading
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date BETWEEN :week_ago AND :today AND s.so_name=:so
            GROUP BY s.scheme_name, b.reading_date
        """), conn, params={"week_ago": week_ago, "today": today, "so": so_name})

    if not last_week_readings.empty:
        pivot_chart = last_week_readings.pivot(index="reading_date", columns="scheme_name", values="total_reading").fillna(0)
        st.subheader("Last 7 days readings by schemes")
        st.line_chart(pivot_chart)
    else:
        st.info("No readings in last 7 days.")

