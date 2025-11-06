# jjm_demo_app.py
# Streamlit Dashboard for Jal Jeevan Mission (JJM)
# Section Officer — Jalmitra Performance Dashboard
# Features:
# - Demo data generator
# - Ranking of Top 10 & Worst 10 Jalmitras (7-day performance)
# - Performance = (days updated + total water supplied)
# - Robust DB query handling with retry on lock

import streamlit as st
import pandas as pd
import datetime
import random
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
import matplotlib.pyplot as plt
from sqlalchemy.pool import SingletonThreadPool

# --- Streamlit Page Config ---
st.set_page_config(page_title="JJM Jalmitra Performance Dashboard", layout="wide")
st.title("💧 Jal Jeevan Mission — Jalmitra Performance Dashboard (SO Role)")
st.markdown("---")

# --- Database Setup ---
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(
    f"sqlite:///{DB_FILE}",
    connect_args={"check_same_thread": False},
    poolclass=SingletonThreadPool
)

# Create tables if not exist
with engine.connect() as conn:
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

# --- DEMO DATA MANAGEMENT ---
st.header("🧪 Demo Data Management")
col1, col2 = st.columns(2)

with col1:
    if st.button("Generate Demo Data"):
        so_name = "SO-Guwahati"
        schemes_list = [f"Scheme {chr(65+i)}" for i in range(20)]  # A–T
        jalmitras = [f"JM-{i+1}" for i in range(20)]

        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))

            # Insert demo schemes
            for scheme in schemes_list:
                functionality = random.choice(["Functional", "Non-Functional"])
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:s, :f, :so)
                """), {"s": scheme, "f": functionality, "so": so_name})

            # Fetch scheme IDs
            schemes_df = pd.read_sql("SELECT * FROM schemes", conn)

            # Insert demo readings for last 7 days
            today = datetime.date.today()
            readings = [110010, 215870, 150340, 189420, 200015, 234870]

            for idx, row in schemes_df.iterrows():
                if row["functionality"] == "Functional":
                    jalmitra = jalmitras[idx % len(jalmitras)]
                    for d in range(7):
                        date = (today - datetime.timedelta(days=d)).isoformat()
                        if random.random() < 0.85:  # 85% chance of update
                            conn.execute(text("""
                                INSERT INTO bfm_readings (scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                                VALUES (:sid, :jm, :r, :dt, :tm, :qty)
                            """), {
                                "sid": row["id"],
                                "jm": jalmitra,
                                "r": random.choice(readings),
                                "dt": date,
                                "tm": f"{random.randint(6,18)}:{random.choice(['00','30'])}:00",
                                "qty": round(random.uniform(50.0, 350.0), 2)
                            })
        st.success("✅ Demo data generated successfully!")

with col2:
    if st.button("Remove Demo Data"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))
        st.warning("🗑️ All demo data removed.")

st.markdown("---")

# --- ROLE SELECTION ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("This page is for Section Officer role only.")
    st.stop()

so_name = "SO-Guwahati"

# --- FETCH SCHEMES ---
with engine.connect() as conn:
    schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name=:so"), conn, params={"so": so_name})

if schemes.empty:
    st.info("No schemes found. Generate demo data first.")
    st.stop()

functional_schemes = schemes[schemes["functionality"] == "Functional"]
if functional_schemes.empty:
    st.warning("No functional schemes found.")
    st.stop()

# --- FETCH LAST 7 DAYS READINGS (with retry) ---
today = datetime.date.today()
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today.isoformat()

query = text("""
    SELECT s.scheme_name, s.id AS scheme_id, b.jalmitra, b.reading_date, b.water_quantity
    FROM bfm_readings b
    JOIN schemes s ON b.scheme_id = s.id
    WHERE b.reading_date BETWEEN :start AND :end
      AND s.so_name = :so
      AND s.functionality = 'Functional'
""")

max_attempts = 5
last7 = pd.DataFrame()
for attempt in range(max_attempts):
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"start": start_date, "end": end_date, "so": so_name})
            rows = result.mappings().all()
            last7 = pd.DataFrame(rows)
        break
    except OperationalError as e:
        if "locked" in str(e).lower() and attempt < max_attempts - 1:
            time.sleep(0.5 * (attempt + 1))
            continue
        else:
            st.error("Database query failed. Check logs for details.")
            raise

if last7.empty:
    st.info("No readings recorded in last 7 days.")
    st.stop()

# --- PERFORMANCE CALCULATION ---
metrics = last7.groupby("jalmitra").agg(
    days_updated=("reading_date", lambda x: x.nunique()),
    total_water=("water_quantity", "sum")
).reset_index()

# Normalize & score
metrics["days_norm"] = metrics["days_updated"] / 7
max_qty = metrics["total_water"].max()
metrics["qty_norm"] = metrics["total_water"] / max_qty if max_qty > 0 else 0
weight_freq, weight_qty = 0.5, 0.5
metrics["score"] = 0.5 * metrics["days_norm"] + 0.5 * metrics["qty_norm"]

# Rank
metrics = metrics.sort_values("score", ascending=False).reset_index(drop=True)
metrics["Rank"] = metrics.index + 1
metrics["total_water"] = metrics["total_water"].round(2)
metrics["score"] = metrics["score"].round(3)

# --- TOP & WORST 10 ---
top_n = 10
top_table = metrics.head(top_n).copy()
worst_table = metrics.tail(top_n).sort_values("score", ascending=True).copy()

top_table.columns = ["Jalmitra", "Days Updated", "Total Water (m³)", "Days Norm", "Qty Norm", "Score", "Rank"]
worst_table.columns = ["Jalmitra", "Days Updated", "Total Water (m³)", "Days Norm", "Qty Norm", "Score", "Rank"]

# --- STYLING ---
def style_table(df, color):
    cmap = "Greens" if color == "green" else "Reds"
    return (
        df.style
        .background_gradient(subset=["Days Updated", "Total Water (m³)", "Score"], cmap=cmap)
        .format({"Total Water (m³)": "{:,.2f}", "Score": "{:.3f}"})
        .set_table_styles([{"selector": "th", "props": [("font-weight", "600")]}])
    )

# --- DISPLAY ---
st.header(f"📊 Jalmitra Performance Ranking (Last 7 Days) — {so_name}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Top Performing Jalmitras")
    st.dataframe(style_table(top_table[["Rank", "Jalmitra", "Days Updated", "Total Water (m³)", "Score"]], "green"), height=420)
    st.download_button("⬇️ Download Top 10", top_table.to_csv(index=False), "top_jalmitras.csv")

with col2:
    st.markdown("### 🔴 Worst Performing Jalmitras")
    st.dataframe(style_table(worst_table[["Rank", "Jalmitra", "Days Updated", "Total Water (m³)", "Score"]], "red"), height=420)
    st.download_button("⬇️ Download Worst 10", worst_table.to_csv(index=False), "worst_jalmitras.csv")

st.markdown("---")
with st.expander("ℹ️ How the Ranking Works"):
    st.markdown("""
    **Performance Score Formula:**
    ```
    score = 0.5 * (days_updated / 7) + 0.5 * (total_water / max_total_water)
    ```
    - Equal weight to both *frequency* and *quantity*  
    - Ranks are based on score (higher = better)
    - Tables show **Top 10 (green)** and **Bottom 10 (red)**
    """)

st.success(f"Rankings generated for last 7 days ({start_date} → {end_date}).")

