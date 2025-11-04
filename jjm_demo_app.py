# jjm_demo_app.py
# Streamlit dashboard for Jal Jeevan Mission — SO Role
# Features:
# - Landing page with role selection
# - Demo data generator & remover
# - Functional/Non-functional schemes
# - BFM readings & water quantity per Jalmitra
# - Last 7 days water supplied chart

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from sqlalchemy import create_engine, text

# --- Page Config ---
st.set_page_config(page_title="JJM Role Dashboard", layout="wide")
st.image("logo.jpg", width=180)
st.title("Jal Jeevan Mission — Landing Dashboard")
st.markdown("---")

# --- Database setup (SQLite) ---
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})

# Create tables if not exist
with engine.connect() as conn:
    conn.execute(text("""
    CREATE TABLE IF NOT EXISTS schemes (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scheme_name TEXT,
        functionality TEXT,  -- Functional / Non-Functional
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

# --- Demo Data Generator / Remover ---
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Demo Data"):
        so_name = "SO-Guwahati"
        schemes_list = [f"Scheme {chr(65+i)}" for i in range(20)]  # Scheme A-T
        jalmitras = [f"JM-{i+1}" for i in range(20)]  # One Jalmitra per scheme

        with engine.begin() as conn:
            # Clear old data
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))

            # Insert schemes
            for i, scheme in enumerate(schemes_list):
                functionality = random.choice(["Functional", "Non-Functional"])
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:scheme_name, :functionality, :so_name)
                """), {"scheme_name": scheme, "functionality": functionality, "so_name": so_name})

            # Fetch scheme IDs
            schemes_df = pd.read_sql("SELECT * FROM schemes", conn)

            # Generate readings only for Functional schemes
            today = datetime.date.today()
            random_readings = [110010, 215870, 150340, 189420, 200015, 234870]
            for _, row in schemes_df.iterrows():
                if row["functionality"] == "Functional":
                    jalmitra = jalmitras[_ % len(jalmitras)]
                    for d in range(7):  # last 7 days
                        date = (today - datetime.timedelta(days=d)).isoformat()
                        reading = random.choice(random_readings)
                        water_qty = round(random.uniform(40.0, 200.0), 2)
                        time = f"{random.randint(6,18)}:{random.choice(['00','30'])}:00"
                        conn.execute(text("""
                            INSERT INTO bfm_readings (scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                            VALUES (:scheme_id, :jalmitra, :reading, :reading_date, :reading_time, :water_quantity)
                        """), {
                            "scheme_id": row["id"],
                            "jalmitra": jalmitra,
                            "reading": reading,
                            "reading_date": date,
                            "reading_time": time,
                            "water_quantity": water_qty
                        })
        st.success("✅ 20 demo schemes and 7 days of readings generated successfully!")

with col2:
    if st.button("Remove Demo Data"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))
        st.warning("🗑️ All demo data removed successfully!")

st.markdown("---")

# --- Role Selection ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

if role == "Section Officer":
    st.header("Section Officer Dashboard")

    so_name = "SO-Guwahati"
    with engine.connect() as conn:
        schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name=:so"), conn, params={"so": so_name})

    st.subheader("All Schemes under SO")
    st.dataframe(schemes)

    # Functional schemes
    functional_schemes = schemes[schemes['functionality'] == "Functional"]
    st.subheader("Functional Schemes under SO")
    st.dataframe(functional_schemes)

    today = datetime.date.today().isoformat()

    # --- Today's readings (Functional schemes only) ---
    with engine.connect() as conn:
        readings_today = pd.read_sql(text("""
            SELECT s.scheme_name, b.jalmitra, b.reading, b.reading_time, b.water_quantity
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date = :today
            AND s.functionality='Functional'
            AND s.so_name=:so
        """), conn, params={"today": today, "so": so_name})

    st.subheader("BFM Readings by Jalmitras Today")
    st.write(f"Total readings recorded today: {len(readings_today)}")
    if not readings_today.empty:
        st.dataframe(readings_today)
    else:
        st.info("No readings recorded today.")

    # --- Water quantity matrix ---
    if not readings_today.empty:
        quantity_matrix = readings_today.pivot_table(index="jalmitra", columns="scheme_name", values="water_quantity", aggfunc="sum").fillna(0)
        st.subheader("💧 Water Quantity Supplied (m³) per Jalmitra per Scheme")
        st.dataframe(quantity_matrix)
    else:
        st.info("No water quantity data available.")

    # --- Absent Jalmitras ---
    all_jalmitras = [f"JM-{i+1}" for i in range(20)]
    absent_list = []
    for j in all_jalmitras:
        for s_name in functional_schemes["scheme_name"]:
            if readings_today.empty or not ((readings_today["jalmitra"] == j) & (readings_today["scheme_name"] == s_name)).any():
                absent_list.append({"jalmitra": j, "scheme_name": s_name})
    absent_df = pd.DataFrame(absent_list)
    st.subheader("Absent Readings by Jalmitras")
    st.dataframe(absent_df)

    # --- Last 7 Days — Water Quantity Graph ---
    week_ago = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    with engine.connect() as conn:
        last_week_qty = pd.read_sql(text("""
            SELECT s.scheme_name, b.reading_date, SUM(b.water_quantity) AS total_water_m3
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date BETWEEN :week_ago AND :today
            AND s.so_name = :so
            AND s.functionality = 'Functional'
            GROUP BY s.scheme_name, b.reading_date
        """), conn, params={"week_ago": week_ago, "today": today, "so": so_name})

    if not last_week_qty.empty:
        pivot_chart = last_week_qty.pivot(index="reading_date", columns="scheme_name", values="total_water_m3").fillna(0)
        st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")
        st.line_chart(pivot_chart)
    else:
        st.info("No data found for the past 7 days.")
