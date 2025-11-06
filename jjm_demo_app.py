# jjm_demo_app.py
# Streamlit dashboard for Jal Jeevan Mission — SO Role
# Updated: Show Top 10 performing Jalmitras (green) and Worst 10 Jalmitras (red)
# Performance = combination of (a) days with readings in last 7 days and (b) cumulative water quantity in last 7 days.
# Score calculation: both parameters normalized to [0,1], equal weight (0.5 each). Higher score = better performer.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
from sqlalchemy import create_engine, text
import matplotlib.pyplot as plt

st.set_page_config(page_title="JJM — Top/Worst Jalmitras (7 days)", layout="wide")
st.title("JJM — Section Officer: Top / Worst Jalmitras (Last 7 Days)")
st.markdown("Shows top-performing and worst-performing Jalmitras based on update frequency and cumulative water quantity.")

# --- DB setup (SQLite) ---
DB_FILE = "jjm_demo.sqlite"
engine = create_engine(f"sqlite:///{DB_FILE}", connect_args={"check_same_thread": False})

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

# --- Demo data generator / remover ---
st.markdown("### 🧪 Demo data")
colg, colr = st.columns(2)

with colg:
    if st.button("Generate Demo Data (20 schemes, 7 days)"):
        so_name = "SO-Guwahati"
        schemes_list = [f"Scheme {chr(65+i)}" for i in range(20)]  # A-T
        jalmitras = [f"JM-{i+1}" for i in range(20)]  # one per scheme mapping

        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))

            for i, scheme in enumerate(schemes_list):
                functionality = random.choice(["Functional", "Non-Functional"])
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:scheme_name, :functionality, :so_name)
                """), {"scheme_name": scheme, "functionality": functionality, "so_name": so_name})

            schemes_df = pd.read_sql("SELECT * FROM schemes", conn)
            today = datetime.date.today()
            random_readings = [110010, 215870, 150340, 189420, 200015, 234870]

            # create 7 days readings only for Functional schemes
            for idx, row in schemes_df.iterrows():
                if row["functionality"] == "Functional":
                    # map scheme index to a jalmitra deterministically
                    jalmitra = jalmitras[idx % len(jalmitras)]
                    # To create variety: some jalmitras may miss days
                    for d in range(7):
                        date = (today - datetime.timedelta(days=d)).isoformat()
                        # simulate some missing days randomly
                        if random.random() < 0.85:  # 85% chance a reading exists that day
                            reading = random.choice(random_readings)
                            water_qty = round(random.uniform(10.0, 350.0), 2)  # wider range
                            time = f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00"
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
        st.success("Demo data generated: 20 schemes and up to 7 days of readings for Functional schemes.")

with colr:
    if st.button("Remove Demo Data"):
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))
        st.warning("All demo data removed.")

st.markdown("---")

# --- Role selection (keeps the original pattern) ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("This dashboard view is for Section Officer role. Choose 'Section Officer' to see the rankings.")
    st.stop()

so_name = "SO-Guwahati"

# --- Fetch functional schemes for this SO ---
with engine.connect() as conn:
    schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name = :so"), conn, params={"so": so_name})

if schemes.empty:
    st.info("No schemes found for this SO. Generate demo data to see results.")
    st.stop()

functional_schemes = schemes[schemes['functionality'] == "Functional"]
if functional_schemes.empty:
    st.info("No functional schemes found. Rankings require functional schemes with readings.")
    st.stop()

# --- Compute last 7 days window ---
today_date = datetime.date.today()
start_date = (today_date - datetime.timedelta(days=6)).isoformat()
end_date = today_date.isoformat()

# --- Query bfm_readings for last 7 days for functional schemes under this SO ---
with engine.connect() as conn:
    last7 = pd.read_sql(text("""
        SELECT s.scheme_name, s.id AS scheme_id, b.jalmitra, b.reading_date, b.water_quantity
        FROM bfm_readings b
        JOIN schemes s ON b.scheme_id = s.id
        WHERE b.reading_date BETWEEN :start AND :end
        AND s.so_name = :so
        AND s.functionality = 'Functional'
    """), conn, params={"start": start_date, "end": end_date, "so": so_name})

# If no readings in the last 7 days
if last7.empty:
    st.info("No readings recorded in the last 7 days for functional schemes. Generate demo data or wait for updates.")
    st.stop()

# --- Performance metrics per Jalmitra ---
# We need:
#  - days_updated: number of distinct days in last 7 with at least one reading
#  - total_water_m3: sum of water_quantity in last 7 days
metrics = last7.groupby("jalmitra").agg(
    days_updated = ("reading_date", lambda x: x.nunique()),
    total_water_m3 = ("water_quantity", "sum"),
    schemes_covered = ("scheme_id", lambda x: x.nunique())
).reset_index()

# Ensure all jalmitras expected (one per functional scheme mapping) appear even if 0 readings
# Extract expected jalmitra list from demo mapping heuristic: "JM-i"
# But safer: collect jalmitras present in any readings OR generate from count of functional schemes
expected_count = len(functional_schemes)
# If there are fewer or more jalmitras in metrics vs expected, we still rely on recorded jalmitras only.
# (This avoids introducing fake zeros for jalmitras not yet seen.)

# --- Normalize and compute score ---
# Score combines days_updated (0..7) and total_water_m3 (0..max)
# Both normalized to 0-1. Equal weight used: score = 0.5 * (days_updated_norm) + 0.5 * (qty_norm)
# If you prefer different weights, adjust weight_freq and weight_qty.
weight_freq = 0.5
weight_qty = 0.5

metrics['days_updated'] = metrics['days_updated'].astype(int)
max_days = 7.0
metrics['days_norm'] = metrics['days_updated'] / max_days

max_qty = metrics['total_water_m3'].max()
if max_qty <= 0:
    # avoid division by zero; if all zero, qty_norm = 0
    metrics['qty_norm'] = 0.0
else:
    metrics['qty_norm'] = metrics['total_water_m3'] / max_qty

metrics['score'] = metrics['days_norm'] * weight_freq + metrics['qty_norm'] * weight_qty

# Additional sorting / rounding for display
metrics = metrics.sort_values(by=['score', 'total_water_m3'], ascending=False).reset_index(drop=True)
metrics['Rank'] = metrics.index + 1
metrics['total_water_m3'] = metrics['total_water_m3'].round(2)
metrics['score'] = metrics['score'].round(3)

# --- Prepare Top and Worst tables ---
top_n = 10
top_table = metrics.sort_values(by='score', ascending=False).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

worst_table = metrics.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
# For worst, recompute rank relative to full list (we'll show their overall Rank from metrics)
# Keep worst sorted from worst -> better (ascending score)
worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

# --- Styling helpers ---
def style_top(df: pd.DataFrame):
    # Use green gradient on Score and Total Water and Days columns
    sty = df.style.format({
        'Total Water (m³)': '{:,.2f}',
        'Score': '{:.3f}'
    })
    # apply green gradient to numeric columns
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
    # bold header
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

def style_worst(df: pd.DataFrame):
    sty = df.style.format({
        'Total Water (m³)': '{:,.2f}',
        'Score': '{:.3f}'
    })
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds')
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

# --- Display side-by-side ---
st.subheader(f"Top {top_n} Performers vs Worst {top_n} (last 7 days) — SO: {so_name}")
col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟢 Top Performers (Best → Worst)")
    if top_table.empty:
        st.info("No top performers to show.")
    else:
        # show best at top (Rank 1 on top)
        st.write(f"Showing top {len(top_table)} Jalmitras ordered by score (higher = better).")
        st.dataframe(style_top(top_table), height=420)

    # small export button
    csv_top = top_table.to_csv(index=False)
    st.download_button("Download Top as CSV", csv_top, file_name="jjm_top_jalmitras.csv", mime="text/csv")

with col2:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to show.")
    else:
        st.write(f"Showing worst {len(worst_table)} Jalmitras ordered by score (lower = worse).")
        st.dataframe(style_worst(worst_table), height=420)

    csv_worst = worst_table.to_csv(index=False)
    st.download_button("Download Worst as CSV", csv_worst, file_name="jjm_worst_jalmitras.csv", mime="text/csv")

st.markdown("---")

# --- Explanation of scoring (small help text) ---
with st.expander("How ranking is computed (click to expand)"):
    st.markdown("""
    - For each Jalmitra, we compute:
      - **Days Updated (last 7d)** — number of distinct days in the last 7 days where at least one reading was submitted (0–7).
      - **Total Water (m³)** — sum of `water_quantity` for the last 7 days.
    - Both metrics are normalized to [0, 1]:
      - `days_norm = days_updated / 7`
      - `qty_norm = total_water_m3 / max_total_water_among_jalmitras`
    - Combined score = `0.5 * days_norm + 0.5 * qty_norm` (equal weighting).
    - Sort descending by `score` for Top performers, ascending for Worst performers.
    - You can change weights in the code (`weight_freq`, `weight_qty`) if you want frequency to matter more or less.
    """)

# --- Show raw metrics (optional toggle) ---
if st.checkbox("Show raw metrics table (all Jalmitras)"):
    st.dataframe(metrics[['Rank','jalmitra','days_updated','total_water_m3','schemes_covered','score']].rename(columns={
        'jalmitra':'Jalmitra',
        'days_updated':'Days Updated (last 7d)',
        'total_water_m3':'Total Water (m³)',
        'schemes_covered':'Schemes Covered'
    }))

st.success("Rankings computed for last 7 days (from {} to {}).".format(start_date, end_date))
