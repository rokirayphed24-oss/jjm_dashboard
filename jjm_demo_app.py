# jjm_demo_app.py
# Streamlit Dashboard (session-state backed) for Jal Jeevan Mission — Jalmitra Performance
# No SQLite — uses st.session_state to avoid OperationalError / DB locks on hosted platforms.

import streamlit as st
import pandas as pd
import datetime
import random
import io

# --- Page config ---
st.set_page_config(page_title="JJM Jalmitra Performance (session)", layout="wide")
st.title("💧 Jal Jeevan Mission — Jalmitra Performance Dashboard (SO Role)")
st.markdown("Data stored in this session (no SQLite) — ideal for demo & debugging on Streamlit Cloud.")
st.markdown("---")

# --- Helpers to initialize session state ---
def init_session_state():
    if "schemes" not in st.session_state:
        st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    if "readings" not in st.session_state:
        st.session_state["readings"] = pd.DataFrame(columns=[
            "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
        ])
    if "next_scheme_id" not in st.session_state:
        st.session_state["next_scheme_id"] = 1
    if "next_reading_id" not in st.session_state:
        st.session_state["next_reading_id"] = 1

init_session_state()

# --- Utility: safe dataframe display with pandas Styler for Streamlit ---
def show_styled_dataframe(st_obj, styler, height=None):
    # Streamlit supports st.dataframe(styler) directly
    st_obj.dataframe(styler, height=height)

# --- Demo data management (session-state based) ---
st.header("🧪 Demo Data Management")
col1, col2 = st.columns(2)

SO_NAME = "SO-Guwahati"
TOTAL_SCHEMES = 20
JALMITRA_PREFIX = "JM-"

with col1:
    if st.button("Generate Demo Data (session)"):
        # create schemes A..T
        schemes_list = [f"Scheme {chr(65+i)}" for i in range(TOTAL_SCHEMES)]
        jalmitras = [f"{JALMITRA_PREFIX}{i+1}" for i in range(TOTAL_SCHEMES)]

        schemes_rows = []
        for scheme in schemes_list:
            func = random.choice(["Functional", "Non-Functional"])
            scheme_id = st.session_state["next_scheme_id"]
            schemes_rows.append({
                "id": scheme_id,
                "scheme_name": scheme,
                "functionality": func,
                "so_name": SO_NAME
            })
            st.session_state["next_scheme_id"] += 1

        st.session_state["schemes"] = pd.DataFrame(schemes_rows)

        # Fill readings (functional schemes only) for last 7 days
        readings_rows = []
        today = datetime.date.today()
        reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

        # Use deterministic mapping of scheme index -> jalmitra
        schemes_df = st.session_state["schemes"]
        for idx, row in schemes_df.reset_index().iterrows():
            if row["functionality"] != "Functional":
                continue
            scheme_id = row["id"]
            jalmitra = jalmitras[idx % len(jalmitras)]
            for d in range(7):
                date = (today - datetime.timedelta(days=d)).isoformat()
                # 85% chance of update that day to simulate real data
                if random.random() < 0.85:
                    rid = st.session_state["next_reading_id"]
                    readings_rows.append({
                        "id": rid,
                        "scheme_id": scheme_id,
                        "jalmitra": jalmitra,
                        "reading": random.choice(reading_samples),
                        "reading_date": date,
                        "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                        "water_quantity": round(random.uniform(50.0, 350.0), 2)
                    })
                    st.session_state["next_reading_id"] += 1

        st.session_state["readings"] = pd.DataFrame(readings_rows)
        st.success("✅ Demo data generated in session (schemes + up to 7 days readings).")

with col2:
    if st.button("Remove Demo Data (session)"):
        # clear session state tables
        st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
        st.session_state["readings"] = pd.DataFrame(columns=[
            "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
        ])
        st.session_state["next_scheme_id"] = 1
        st.session_state["next_reading_id"] = 1
        st.warning("🗑️ Session demo data removed.")

st.markdown("---")

# --- Role selection ---
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.info("Switch to 'Section Officer' to view Jalmitra rankings.")
    st.stop()

# --- Fetch session data for SO ---
schemes = st.session_state["schemes"]
readings = st.session_state["readings"]

if schemes.empty:
    st.info("No schemes available. Generate demo data to proceed.")
    st.stop()

# Filter schemes for this SO
schemes_so = schemes[schemes["so_name"] == SO_NAME] if "so_name" in schemes.columns else schemes
functional_schemes = schemes_so[schemes_so["functionality"] == "Functional"]

if functional_schemes.empty:
    st.info("No Functional schemes found under this SO. Generate demo data with functional schemes.")
    st.stop()

# --- Compute last 7 days window & metrics from session tables ---
today = datetime.date.today()
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today.isoformat()

# Filter readings in the last 7 days for functional schemes under this SO
if readings.empty:
    st.info("No readings recorded in the session yet. Generate demo data to see example readings.")
    st.stop()

# join readings with scheme info using pandas merge
merged = readings.merge(functional_schemes[["id", "scheme_name"]], left_on="scheme_id", right_on="id", how="inner", suffixes=("", "_scheme"))
if merged.empty:
    st.info("No readings found for functional schemes in last 7 days.")
    st.stop()

# filter date range
mask = (merged["reading_date"] >= start_date) & (merged["reading_date"] <= end_date)
last7 = merged.loc[mask].copy()

if last7.empty:
    st.info("No readings in the last 7 days for functional schemes.")
    st.stop()

# --- Compute performance metrics per Jalmitra ---
metrics = last7.groupby("jalmitra").agg(
    days_updated = ("reading_date", lambda x: x.nunique()),
    total_water_m3 = ("water_quantity", "sum"),
    schemes_covered = ("scheme_id", lambda x: x.nunique())
).reset_index()

# ensure numeric types
metrics["days_updated"] = metrics["days_updated"].astype(int)
metrics["total_water_m3"] = metrics["total_water_m3"].astype(float)

# normalize and compute score
max_days = 7.0
metrics["days_norm"] = metrics["days_updated"] / max_days
max_qty = metrics["total_water_m3"].max() if not metrics["total_water_m3"].empty else 0.0
metrics["qty_norm"] = metrics["total_water_m3"] / max_qty if max_qty > 0 else 0.0

# default weights (you can later make UI sliders)
weight_freq = 0.5
weight_qty = 0.5
metrics["score"] = metrics["days_norm"] * weight_freq + metrics["qty_norm"] * weight_qty

# ranking and formatting
metrics = metrics.sort_values(by=["score", "total_water_m3"], ascending=False).reset_index(drop=True)
metrics["Rank"] = metrics.index + 1
metrics["total_water_m3"] = metrics["total_water_m3"].round(2)
metrics["score"] = metrics["score"].round(3)

# --- Prepare top and worst tables (10 each) ---
top_n = 10
top_table = metrics.sort_values(by="score", ascending=False).head(top_n)[["Rank","jalmitra","days_updated","total_water_m3","score"]].copy()
top_table.columns = ["Rank","Jalmitra","Days Updated (last 7d)","Total Water (m³)","Score"]

worst_table = metrics.sort_values(by="score", ascending=True).head(top_n)[["Rank","jalmitra","days_updated","total_water_m3","score"]].copy()
worst_table.columns = ["Rank","Jalmitra","Days Updated (last 7d)","Total Water (m³)","Score"]

# --- Styling functions using pandas Styler ---
def style_top(df: pd.DataFrame):
    sty = df.style.format({
        "Total Water (m³)": "{:,.2f}",
        "Score": "{:.3f}"
    })
    # green gradient for numeric cols
    sty = sty.background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Greens")
    sty = sty.set_table_styles([{"selector":"th", "props":[("font-weight","600")]}])
    return sty

def style_worst(df: pd.DataFrame):
    sty = df.style.format({
        "Total Water (m³)": "{:,.2f}",
        "Score": "{:.3f}"
    })
    sty = sty.background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Reds")
    sty = sty.set_table_styles([{"selector":"th", "props":[("font-weight","600")]}])
    return sty

# --- Display side-by-side ---
st.subheader(f"Top {top_n} Performers vs Worst {top_n} (Last 7 Days) — SO: {SO_NAME}")
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🟢 Top Performers (Best → Worst)")
    if top_table.empty:
        st.info("No top performers to display.")
    else:
        # show styled dataframe
        st.dataframe(style_top(top_table), height=420)
    # CSV download
    csv_buf = top_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Top 10 CSV", csv_buf, file_name="jjm_top_10.csv", mime="text/csv")

with col_right:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to display.")
    else:
        st.dataframe(style_worst(worst_table), height=420)
    csv_buf2 = worst_table.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Worst 10 CSV", csv_buf2, file_name="jjm_worst_10.csv", mime="text/csv")

st.markdown("---")

with st.expander("ℹ️ How ranking is computed"):
    st.markdown("""
    - **Days Updated (last 7d)**: distinct days (0–7) a Jalmitra submitted at least one reading.
    - **Total Water (m³)**: cumulative `water_quantity` for the last 7 days.
    - Both metrics normalized to [0,1] and combined:
      `score = 0.5 * days_norm + 0.5 * qty_norm`.
    - Higher score = better performer.
    - Data is stored only for the current session (no SQLite). To keep data across restarts, we can later add a persistent store.
    """)

st.success(f"Rankings generated for last 7 days ({start_date} → {end_date}).")
