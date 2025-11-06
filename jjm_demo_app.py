# jjm_demo_app.py
# Unified Streamlit app: full original dashboard + Top/Worst Jalmitra performance
# - Uses st.session_state (no SQLite) for stability on Streamlit Cloud
# - Includes SO / AEE / EE role selection (AEE/EE are placeholders)
# - All original views: schemes, functional/non-functional, today's readings,
#   participation pie, 3-column water quantity table, 7-day line chart
# - New: Top/Worst Jalmitra ranking (last 7 days), CSV downloads
# Save as jjm_demo_app.py and run: streamlit run jjm_demo_app.py

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import matplotlib.pyplot as plt
import io

# ---------------------------
# Page config & header
# ---------------------------
st.set_page_config(page_title="JJM Role Dashboard (Unified)", layout="wide")
st.image("logo.jpg", width=160, clamp=False) if True else None  # ignore error if missing
st.title("Jal Jeevan Mission — Role Dashboard (Unified)")
st.markdown("Manage demo data, view schemes & readings, and see Jalmitra performance rankings (last 7 days).")
st.markdown("---")

# ---------------------------
# Session-state initialization
# ---------------------------
def init_state():
    if "schemes" not in st.session_state:
        st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    if "readings" not in st.session_state:
        st.session_state["readings"] = pd.DataFrame(columns=[
            "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
        ])
    # deterministic jalmitra list used for mapping JM-1..JM-N
    if "jalmitras" not in st.session_state:
        st.session_state["jalmitras"] = []
    if "next_scheme_id" not in st.session_state:
        st.session_state["next_scheme_id"] = 1
    if "next_reading_id" not in st.session_state:
        st.session_state["next_reading_id"] = 1
    # track that demo data was generated at least once
    if "demo_generated" not in st.session_state:
        st.session_state["demo_generated"] = False

init_state()

# ---------------------------
# Helpers
# ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id", "scheme_name", "functionality", "so_name"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity"
    ])
    st.session_state["jalmitras"] = []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False

def generate_demo_data(total_schemes=20, so_name="SO-Guwahati", update_prob=0.85):
    """
    Generates demo schemes (A..), maps JM-1..JM-N one per scheme, and
    generates up to 7 days of readings for Functional schemes only.
    """
    schemes_rows = []
    jalmitras = [f"JM-{i+1}" for i in range(total_schemes)]
    today = datetime.date.today()
    reading_samples = [110010, 215870, 150340, 189420, 200015, 234870]

    # Insert schemes
    for i in range(total_schemes):
        scheme_name = f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}"
        functionality = random.choice(["Functional", "Non-Functional"])
        scheme_id = st.session_state["next_scheme_id"]
        schemes_rows.append({
            "id": scheme_id,
            "scheme_name": scheme_name,
            "functionality": functionality,
            "so_name": so_name
        })
        st.session_state["next_scheme_id"] += 1

    st.session_state["schemes"] = pd.DataFrame(schemes_rows)
    st.session_state["jalmitras"] = jalmitras

    # Insert readings for Functional schemes only
    readings_rows = []
    for idx, row in st.session_state["schemes"].reset_index().iterrows():
        if row["functionality"] != "Functional":
            continue
        scheme_id = row["id"]
        jalmitra = jalmitras[idx % len(jalmitras)]
        for d in range(7):
            date = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < update_prob:
                rid = st.session_state["next_reading_id"]
                readings_rows.append({
                    "id": rid,
                    "scheme_id": scheme_id,
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity": round(random.uniform(40.0, 350.0), 2)
                })
                st.session_state["next_reading_id"] += 1

    st.session_state["readings"] = pd.DataFrame(readings_rows)
    st.session_state["demo_generated"] = True

# ---------------------------
# Demo Data Management UI
# ---------------------------
st.markdown("### 🧪 Demo Data Management (session-backed)")
col_gen, col_rem = st.columns([1,1])

with col_gen:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=100, value=20, step=1)
    if st.button("Generate Demo Data"):
        reset_session_data()
        generate_demo_data(total_schemes=total_schemes)
        st.success(f"Generated {total_schemes} demo schemes and up to 7 days of readings for functional schemes.")

with col_rem:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("All demo data removed from session.")

st.markdown("---")

# ---------------------------
# Role Selection
# ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])

# ---------------------------
# AEE / EE placeholders
# ---------------------------
if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer (AEE) — Placeholder")
    st.info("AEE dashboard placeholder — tell me if you want same functionality adapted for AEE.")
    st.stop()

if role == "Executive Engineer":
    st.header("Executive Engineer (EE) — Placeholder")
    st.info("EE dashboard placeholder — tell me if you want same functionality adapted for EE.")
    st.stop()

# ---------------------------
# Section Officer Dashboard (main)
# ---------------------------
st.header("Section Officer Dashboard")
so_name = "SO-Guwahati"

# quick references
schemes_df = st.session_state["schemes"].copy()
readings_df = st.session_state["readings"].copy()
jalmitras_list = st.session_state["jalmitras"]

# if no schemes exist
if schemes_df.empty:
    st.info("No schemes found. Generate demo data to populate the dashboard.")
    st.stop()

# show overview pie only after demo generation
if st.session_state["demo_generated"]:
    st.subheader("📊 Overview — Scheme Functionality")
    func_counts = schemes_df['functionality'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(4,4))
    colors = []
    for label in func_counts.index:
        if label == "Functional":
            colors.append('#4CAF50')
        elif label == "Non-Functional":
            colors.append('#F44336')
        else:
            colors.append(None)
    ax1.pie(func_counts, labels=func_counts.index, autopct='%1.0f%%', startangle=90, colors=colors)
    ax1.set_title("Functional vs Non-Functional")
    st.pyplot(fig1)

st.markdown("---")

# All schemes table
st.subheader("All Schemes under SO")
st.dataframe(schemes_df)

# Functional schemes table
functional_schemes = schemes_df[schemes_df['functionality'] == "Functional"]
st.subheader("Functional Schemes under SO")
if functional_schemes.empty:
    st.info("No functional schemes found under this SO.")
else:
    st.dataframe(functional_schemes)

# ---------------------------
# Today's readings (Functional only)
# ---------------------------
today = datetime.date.today().isoformat()
if readings_df.empty:
    st.subheader("BFM Readings by Jalmitras Today")
    st.info("No readings available in session. Generate demo data to create readings.")
else:
    readings_today = readings_df.merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    # filter functional & SO & today's date
    readings_today = readings_today[
        (readings_today['functionality'] == 'Functional') &
        (readings_today['so_name'] == so_name) &
        (readings_today['reading_date'] == today)
    ][['scheme_name','jalmitra','reading','reading_time','water_quantity']]

    st.subheader("BFM Readings by Jalmitras Today")
    st.write(f"Total readings recorded today: **{len(readings_today)}**")
    if not readings_today.empty:
        st.dataframe(readings_today)
    else:
        st.info("No readings recorded today for functional schemes.")

    # Jalmitra participation pie chart (Updated vs Absent) - (4x4)
    total_functional = len(functional_schemes)
    if total_functional > 0:
        updated_count = readings_today['jalmitra'].nunique() if not readings_today.empty else 0
        absent_count = total_functional - updated_count
        if absent_count < 0:
            absent_count = 0
        counts = [updated_count, absent_count]
        labels = [f"Updated ({updated_count})", f"Absent ({absent_count})"]

        fig2, ax2 = plt.subplots(figsize=(4,4))
        def autopct_with_count(pct, allvals):
            absolute = int(round(pct/100.0 * sum(allvals)))
            return f"{pct:.0f}%\n({absolute})"
        ax2.pie(counts, labels=labels, autopct=lambda pct: autopct_with_count(pct, counts), startangle=90, colors=['#4CAF50','#F44336'])
        ax2.set_title("Jalmitra Updates Today")
        st.pyplot(fig2)
    else:
        st.info("No functional schemes — cannot compute Jalmitra participation.")

# ---------------------------
# Water Quantity Summary (3 columns)
# ---------------------------
if not readings_today.empty:
    st.subheader("💧 Water Quantity Supplied (m³) per Jalmitra per Scheme (Today)")
    simple_table = readings_today[['jalmitra','scheme_name','water_quantity']].copy()
    simple_table.columns = ['Jalmitra','Scheme','Water Quantity (m³)']
    st.dataframe(simple_table)
else:
    st.info("No water quantity records to display for today.")

# ---------------------------
# Last 7 Days — Line Chart (Functional schemes)
# ---------------------------
st.markdown("---")
st.subheader("📈 Last 7 Days — Water Supplied (m³) for Functional Schemes")
if readings_df.empty or functional_schemes.empty:
    st.info("No data to show for the past 7 days.")
else:
    # filter readings for functional schemes and last 7 days
    start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    mask = (readings_df['reading_date'] >= start_date) & (readings_df['reading_date'] <= today)
    last7_all = readings_df.loc[mask].merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    last7_all = last7_all[(last7_all['functionality'] == 'Functional') & (last7_all['so_name'] == so_name)]
    if last7_all.empty:
        st.info("No data found for the past 7 days.")
    else:
        last_week_qty = last7_all.groupby(['scheme_name','reading_date'])['water_quantity'].sum().reset_index()
        pivot_chart = last_week_qty.pivot(index='reading_date', columns='scheme_name', values='water_quantity').fillna(0)
        # sort by date ascending
        pivot_chart = pivot_chart.sort_index()
        st.line_chart(pivot_chart)

# ---------------------------
# Top / Worst Jalmitra Rankings (Last 7 days)
# ---------------------------
st.markdown("---")
st.subheader("🏅 Jalmitra Performance — Top & Worst (Last 7 Days)")

# If no last7_all variable or it's empty, compute similarly
if 'last7_all' not in locals() or last7_all.empty:
    # recompute last7_all as above
    if readings_df.empty:
        st.info("No readings recorded — generate demo data to see rankings.")
        st.stop()
    start_date = (datetime.date.today() - datetime.timedelta(days=6)).isoformat()
    mask = (readings_df['reading_date'] >= start_date) & (readings_df['reading_date'] <= today)
    last7_all = readings_df.loc[mask].merge(schemes_df[['id','scheme_name','functionality','so_name']], left_on='scheme_id', right_on='id', how='left')
    last7_all = last7_all[(last7_all['functionality'] == 'Functional') & (last7_all['so_name'] == so_name)]

if last7_all.empty:
    st.info("No readings in the last 7 days for functional schemes — cannot compute rankings.")
    st.stop()

# compute metrics per jalmitra
metrics = last7_all.groupby('jalmitra').agg(
    days_updated = ('reading_date', lambda x: x.nunique()),
    total_water_m3 = ('water_quantity', 'sum'),
    schemes_covered = ('scheme_id', lambda x: x.nunique())
).reset_index()

# Ensure every expected jalmitra (JM-1..JM-N) is present with zero values if not seen
expected_jalmitras = jalmitras_list if jalmitras_list else [f"JM-{i+1}" for i in range(len(schemes_df))]
# Add missing jalmitras with zeros
for jm in expected_jalmitras:
    if jm not in metrics['jalmitra'].values:
        metrics = pd.concat([metrics, pd.DataFrame([{
            'jalmitra': jm,
            'days_updated': 0,
            'total_water_m3': 0.0,
            'schemes_covered': 0
        }])], ignore_index=True)

# numeric conversions
metrics['days_updated'] = metrics['days_updated'].astype(int)
metrics['total_water_m3'] = metrics['total_water_m3'].astype(float)

# normalize & score (equal weights)
max_days = 7.0
metrics['days_norm'] = metrics['days_updated'] / max_days
max_qty = metrics['total_water_m3'].max() if not metrics['total_water_m3'].empty else 0.0
metrics['qty_norm'] = metrics['total_water_m3'] / max_qty if max_qty > 0 else 0.0
weight_freq = 0.5
weight_qty = 0.5
metrics['score'] = (metrics['days_norm'] * weight_freq) + (metrics['qty_norm'] * weight_qty)

# ranking
metrics = metrics.sort_values(by=['score','total_water_m3'], ascending=False).reset_index(drop=True)
metrics['Rank'] = metrics.index + 1
metrics['total_water_m3'] = metrics['total_water_m3'].round(2)
metrics['score'] = metrics['score'].round(3)

# prepare top & worst (10 each)
top_n = 10
top_table = metrics.sort_values(by='score', ascending=False).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
top_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

worst_table = metrics.sort_values(by='score', ascending=True).head(top_n)[['Rank','jalmitra','days_updated','total_water_m3','score']].copy()
worst_table.columns = ['Rank','Jalmitra','Days Updated (last 7d)','Total Water (m³)','Score']

# styling functions (pandas Styler)
def style_top(df):
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Greens')
    sty = sty.set_table_styles([{'selector':'th','props':[('font-weight','600')]}])
    return sty

def style_worst(df):
    sty = df.style.format({'Total Water (m³)': '{:,.2f}', 'Score': '{:.3f}'})
    sty = sty.background_gradient(subset=['Days Updated (last 7d)','Total Water (m³)','Score'], cmap='Reds')
    sty = sty.set_table_styles([{'selector':'th','props':[('font-weight','600')]}])
    return sty

col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### 🟢 Top Performers (Best → Worst)")
    if top_table.empty:
        st.info("No top performers to show.")
    else:
        st.dataframe(style_top(top_table), height=420)
        csv_top = top_table.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Top 10 CSV", csv_top, file_name="jjm_top_10.csv", mime="text/csv")

with col_right:
    st.markdown("### 🔴 Worst Performers (Worst → Better)")
    if worst_table.empty:
        st.info("No worst performers to show.")
    else:
        st.dataframe(style_worst(worst_table), height=420)
        csv_worst = worst_table.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Download Worst 10 CSV", csv_worst, file_name="jjm_worst_10.csv", mime="text/csv")

st.markdown("---")

# Explanation and optional raw metrics view
with st.expander("ℹ️ How ranking is computed"):
    st.markdown("""
    - Days Updated (last 7d): number of distinct days (0–7) a Jalmitra submitted at least one reading.
    - Total Water (m³): cumulative water_quantity for the last 7 days.
    - Normalized: days_norm = days_updated / 7, qty_norm = total_water / max_total_water.
    - Score = 0.5 * days_norm + 0.5 * qty_norm (equal weights).
    - Top table sorts by score descending; Worst sorts by score ascending.
    """)

if st.checkbox("Show full raw metrics table (all Jalmitras)"):
    st.dataframe(metrics[['Rank','jalmitra','days_updated','total_water_m3','schemes_covered','score']].rename(columns={
        'jalmitra':'Jalmitra',
        'days_updated':'Days Updated (last 7d)',
        'total_water_m3':'Total Water (m³)',
        'schemes_covered':'Schemes Covered'
    }))

st.success(f"Dashboard ready. Demo data generated: {st.session_state['demo_generated']}. Data stored in this session only.")
