import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from sqlalchemy import create_engine, text

# --- Page Config ---
st.set_page_config(page_title="JJM Role Dashboard", layout="wide")

st.image("logo.jpg", width=180)
st.title("Jal Jeevan Mission —  Dashboard")
st.markdown("---")

# --- Database setup (simulate SQLite) ---
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})

# Temporary patch: rename 'functional' column to 'functionality' if exists
with engine.connect() as conn:
    try:
        conn.execute(text("ALTER TABLE schemes RENAME COLUMN functional TO functionality"))
        conn.commit()
    except Exception:
        pass  # ignore if already renamed


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

# ------------------------------------------------------------------
# 🔹 DEMO DATA GENERATOR / REMOVER
st.markdown("### 🧪 Demo Data Management")

col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Demo Data"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))

            # Create 20 random schemes
            schemes_data = []
            for i in range(20):
                schemes_data.append({
                    "scheme_name": f"Scheme_{i+1}",
                    "functionality": random.choice(["Functional", "Non-Functional"]),
                    "so_name": "SO-Guwahati"
                })
            for row in schemes_data:
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:scheme_name, :functionality, :so_name)
                """), row)

            # Create random readings for 7 days
            jalmitras = ["JM-1", "JM-2", "JM-3", "JM-4"]
            today = datetime.date.today()
            for day_offset in range(7):
                date_str = (today - datetime.timedelta(days=day_offset)).isoformat()
                for jm in jalmitras:
                    for s_id in range(1, 21):
                        if random.random() > 0.4:  # 60% chance of reading
                            random_reading = random.choice([110010, 215870, 325640, 458920, 562310, 674520])
                            water_quantity = round(random.uniform(5.0, 50.0), 2)
                            time_str = f"{random.randint(6,18)}:{random.randint(0,59):02d}:00"
                            conn.execute(text("""
                                INSERT INTO bfm_readings (scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                                VALUES (:scheme_id, :jalmitra, :reading, :reading_date, :reading_time, :water_quantity)
                            """), {
                                "scheme_id": s_id,
                                "jalmitra": jm,
                                "reading": random_reading,
                                "reading_date": date_str,
                                "reading_time": time_str,
                                "water_quantity": water_quantity
                            })
        st.success("✅ Demo data (20 schemes & readings) generated successfully!")

with col2:
    if st.button("Remove Demo Data"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))
        st.warning("🗑️ All demo data removed successfully!")

st.markdown("---")
# ------------------------------------------------------------------

# --- Role selection ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

if role == "Section Officer":
    st.header("Section Officer Dashboard")

    so_name = "SO-Guwahati"
    with engine.connect() as conn:
        schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name=:so"), conn, params={"so": so_name})

    st.subheader("All Schemes under SO")
    st.dataframe(schemes)

    # Filter Functional schemes
    functional_schemes = schemes[schemes['functionality'] == "Functional"]
    st.subheader("Functional Schemes under SO")
    st.dataframe(functional_schemes)

    today = datetime.date.today().isoformat()

    # --- Today's readings (Functional only)
    with engine.connect() as conn:
        readings_today = pd.read_sql(text("""
            SELECT s.scheme_name, b.jalmitra, b.reading, b.reading_time, b.water_quantity
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date = :today AND s.functionality='Functional' AND s.so_name=:so
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

    # --- Absent Jalmitras (didn't record today)
    all_jalmitras = ["JM-1", "JM-2", "JM-3", "JM-4"]
    absent_list = []
    for j in all_jalmitras:
        for s_name in functional_schemes["scheme_name"]:
            if readings_today.empty or not ((readings_today["jalmitra"] == j) & (readings_today["scheme_name"] == s_name)).any():
                absent_list.append({"jalmitra": j, "scheme_name": s_name})
    absent_df = pd.DataFrame(absent_list)
    st.subheader("Absent Readings by Jalmitras")
    st.dataframe(absent_df)

    # --- Graph: last 7 days readings (Functional schemes only)
    week_ago = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    with engine.connect() as conn:
        last_week_readings = pd.read_sql(text("""
            SELECT s.scheme_name, b.reading_date, SUM(b.reading) as total_reading
            FROM bfm_readings b
            JOIN schemes s ON b.scheme_id = s.id
            WHERE b.reading_date BETWEEN :week_ago AND :today
            AND s.functionality='Functional' AND s.so_name=:so
            GROUP BY s.scheme_name, b.reading_date
        """), conn, params={"week_ago": week_ago, "today": today, "so": so_name})

    if not last_week_readings.empty:
        pivot_chart = last_week_readings.pivot(index="reading_date", columns="scheme_name", values="total_reading").fillna(0)
        st.subheader("📈 Last 7 Days — BFM Readings (Functional Schemes)")
        st.line_chart(pivot_chart)
    else:
        st.info("No readings for the past 7 days.")

