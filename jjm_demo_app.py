# jjm_demo_app.py
# Streamlit Dashboard for Jal Jeevan Mission (JJM)
# Section Officer — Jalmitra Performance Dashboard (Full working code)
# Fixes KeyError during demo data insertion by using explicit SELECT and itertuples()

import streamlit as st
import pandas as pd
import datetime
import random
import time
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError
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

# Ensure tables exist
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
            # clear previous demo data
            conn.execute(text("DELETE FROM schemes"))
            conn.execute(text("DELETE FROM bfm_readings"))

            # Insert demo schemes deterministically (so insert order is preserved)
            for scheme in schemes_list:
                functionality = random.choice(["Functional", "Non-Functional"])
                conn.execute(text("""
                    INSERT INTO schemes (scheme_name, functionality, so_name)
                    VALUES (:sname, :func, :so)
                """), {"sname": scheme, "func": functionality, "so": so_name})

            # Fetch the schemes with explicit columns to ensure 'id' exists
            schemes_df = pd.read_sql("SELECT id, scheme_name, functionality FROM schemes ORDER BY id ASC", conn)

            # Insert demo readings for functional schemes only
            today = datetime.date.today()
            readings_sample = [110010, 215870, 150340, 189420, 200015, 234870]

            # iterate safely using itertuples() to avoid KeyError on column names
            for idx, row in enumerate(schemes_df.itertuples(index=False)):
                # row has attributes: id, scheme_name, functionality
                scheme_id = getattr(row, "id", None)
                functionality = getattr(row, "functionality", None)
                if functionality != "Functional":
                    continue
                # map scheme to a jalmitra deterministically
                jalmitra = jalmitras[idx % len(jalmitras)]
                for d in range(7):
                    date = (today - datetime.timedelta(days=d)).isoformat()
                    # simulate some missing days randomly
                    if random.random() < 0.85:  # 85% chance of an update that day
                        conn.execute(text("""
                            INSERT INTO bfm_readings (scheme_id, jalmitra, reading, reading_date, reading_time, water_quantity)
                            VALUES (:sid, :jm, :r, :dt, :tm, :qty)
                        """), {
                            "sid": scheme_id,
                            "jm": jalmitra,
                            "r": random.choice(readings_sample),
                            "dt": date,
                            "tm": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                            "qty": round(random.uniform(50.0, 350.0), 2)
                        })
        st.success("✅ Demo data generated successfully (20 schemes; up to 7 days readings).")

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
    st.info("This view is for Section Officer role; switch to Section Officer to see rankings.")
    st.stop()

so_name = "SO-Guwahati"

# --- FETCH SCHEMES FOR SO ---
with engine.connect() as conn:
    schemes = pd.read_sql(text("SELECT * FROM schemes WHERE so_name = :so"), conn, params={"so": so_name})

if schemes.empty:
    st.info("No schemes found for this SO. Generate demo data to proceed.")
    st.stop()

functional_schemes = schemes[schemes["functionality"] == "Functional"]
if functional_schemes.empty:
    st.info("No functional schemes found under this SO. Rankings require functional schemes with readings.")
    st.stop()

# --- FETCH LAST 7 DAYS READINGS (robust with retries) ---
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
            rows = result.mappings().all()  # list of dict-like rows
            last7 = pd.DataFrame(rows)
        break
    except OperationalError as e:
        # handle transient sqlite locking
        msg = str(e).lower()
        if "locked" in msg and attempt < max_attempts - 1:
            time.sleep(0.5 * (attempt + 1))
            continue
        else:
            st.error("Database query failed. Check app logs for full details.")
            raise

if last7.empty:
    st.info("No readings recorded in the last 7 days for functional schemes. Generate demo data or wait for updates.")
    st.stop()

# --- COMPUTE PERFORMANCE METRICS PER JALMITRA ---
metrics = last7.groupby("jalmitra").agg(
    days_updated=("reading_date", lambda x: x.nunique()),
    total_water_m3=("water_quantity", "sum"),
    schemes_covered=("scheme_id", lambda x: x.nunique())
).reset_index()

# Ensure numeric types
metrics["days_updated"] = metrics["days_updated"].astype(int)
metrics["total_water_m3"] = metrics["total_water_m3"].astype(float)

# --- Normalize & Score ---
max_days = 7.0
metrics["days_norm"] = metrics["days_updated"] / max_days

max_qty = metrics["total_water_m3"].max()
metrics["qty_norm"] = metrics["total_water_m3"] / max_qty if max_qty and max_qty > 0 else 0.0

# Equal weights by default (tweakable)
weight_freq = 0.5
weight_qty = 0.5
metrics["score"] = metrics["days_norm"] * weight_freq + metrics["qty_norm"] * weight_qty

# --- Rank & Format ---
metrics = metrics.sort_values(by=["score", "total_water_m3"], ascending=False).reset_index(drop=True)
metrics["Rank"] = metrics.index + 1
metrics["total_water_m3"] = metrics["total_water_m3"].round(2)
metrics["score"] = metrics["score"].round(3)

# --- Prepare Top and Worst tables (10 each) ---
top_n = 10
top_table = metrics.sort_values(by="score", ascending=False).head(top_n)[["Rank", "jalmitra", "days_updated", "total_water_m3", "score"]].copy()
top_table.columns = ["Rank", "Jalmitra", "Days Updated (last 7d)", "Total Water (m³)", "Score"]

worst_table = metrics.sort_values(by="score", ascending=True).head(top_n)[["Rank", "jalmitra", "days_updated", "total_water_m3", "score"]].copy()
worst_table.columns = ["Rank", "Jalmitra", "Days Updated (last 7d)", "Total Water (m³)", "Score"]

# --- Styling helpers ---
def style_top(df: pd.DataFrame):
    sty = df.style.format({
        'Total Water (m³)': '{:,.2f}',
        'Score': '{:.3f}'
    })
    sty = sty.background_gradient(subset=['Days Updated (last 7d)', 'Total Water (m³)', 'Score'], cmap='Greens')
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

def style_worst(df: pd.DataFrame):
    sty = df.style.format({
        'Total Water (m³)': '{:,.2f}',
        'Score': '{:.3f}'
    })
    sty = sty.background_gradient(subset=['Days Updated (last 7d)', 'Total Water (m³)', 'Score'], cmap='Reds')
    sty = sty.set_table_styles([{'selector': 'th', 'props': [('font-weight', '600')]}])
    return sty

# --- Display side-by-side ---
st.subheader(f"Top {top_n} Performers vs Worst {top_n} (Last 7 Days) — SO: {so_name}")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🟢 Top Performers (Best → Worst)")
    if top_table.empty:
        st.info("No top performers to show.")
    else:
        st.dataframe(style_top(top_table), height=420)
    st.download_button("Download Top as CSV", top_table.to_csv(index=False), file_name="jjm_top_jalmitras.csv", mime="text/csv")

with col_right:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to show.")
    else:
        st.dataframe(style_worst(worst_table), height=420)
    st.download_button("Download Worst as CSV", worst_table.to_csv(index=False), file_name="jjm_worst_jalmitras.csv", mime="text/csv")

st.markdown("---")
with st.expander("How ranking is computed"):
    st.markdown("""
    - **Days Updated (last 7d)**: number of distinct days (0–7) where a Jalmitra submitted at least one reading.
    - **Total Water (m³)**: cumulative water_quantity over the last 7 days.
    - Normalize both metrics to [0,1], then compute:
      `score = 0.5 * days_norm + 0.5 * qty_norm`
    - Sort descending for Top performers; ascending for Worst performers.
    - Change weights `weight_freq` / `weight_qty` in the code to tweak importance.
    """)

st.success(f"Rankings computed for last 7 days ({start_date} → {end_date}).")
