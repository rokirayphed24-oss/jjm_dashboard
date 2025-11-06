# jjm_demo_app.py
# JJM Dashboard — Final Version
# Updates:
# - In "BFM Readings Updated Today": 
#   -> Removed left margin/blank column
#   -> Reading time moved after BFM Reading
#   -> Times generated mostly in morning and shown in 12-hour format (AM/PM)
# - Everything else same: clickable names, styled tables, phone/web toggle

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Final", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("For Section Officer **ROKI RAY** — Tap a name (below the tables) to view 7-day performance.")
st.markdown("---")

# --------------------------- View mode toggle ---------------------------
view_mode = st.radio("View Mode", ["Web View", "Phone View"], horizontal=True)
st.markdown("---")

# --------------------------- Helpers ---------------------------
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

def init_state():
    st.session_state.setdefault("schemes", pd.DataFrame(columns=["id","scheme_name","functionality","so_name"]))
    st.session_state.setdefault("readings", pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]))
    st.session_state.setdefault("jalmitras", [])
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)

init_state()

# --------------------------- Demo data functions ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    st.session_state["jalmitras"] = []
    st.session_state["demo_generated"] = False
    st.session_state["selected_jalmitra"] = None

def generate_demo_data(total_schemes:int=20, so_name:str="ROKI RAY"):
    FIXED_UPDATE_PROB = 0.85
    assamese = [
        "Biren","Nagen","Rahul","Vikram","Debojit","Anup","Kamal","Ranjit","Himangshu",
        "Pranjal","Rupam","Dilip","Utpal","Amit","Jayanta","Hemanta","Rituraj","Dipankar",
        "Bikash","Dhruba","Subham","Pritam","Saurav","Bijoy","Manoj"
    ]
    jalmitras = random.sample(assamese * 3, total_schemes)
    villages = [
        "Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar",
        "Jorhat","Hajo","Tihu","Kokrajhar","Nalbari","Barpeta","Rangia","Goalpara","Dhemaji",
        "Dibrugarh","Mariani","Sonari"
    ]
    today = datetime.date.today()
    readings = []
    schemes = []
    reading_samples = [110010,215870,150340,189420,200015,234870]

    for i in range(total_schemes):
        schemes.append({
            "id": i+1,
            "scheme_name": f"Scheme {chr(65 + (i % 26))}",
            "functionality": random.choice(["Functional","Non-Functional"]),
            "so_name": so_name
        })

    # mostly morning times (6:00 AM - 11:45 AM)
    for i, s in enumerate(schemes):
        if s["functionality"] != "Functional":
            continue
        scheme_label = random.choice(villages) + " PWSS"
        jalmitra = jalmitras[i % len(jalmitras)]
        for d in range(7):
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                hour = random.randint(6, 11)
                minute = random.choice([0, 15, 30, 45])
                ampm = "AM"
                time_str = f"{hour}:{minute:02d} {ampm}"
                water_qty = round(random.uniform(10.0, 100.0), 2)
                readings.append({
                    "id": len(readings) + 1,
                    "scheme_id": s["id"],
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date_iso,
                    "reading_time": time_str,
                    "water_quantity": water_qty,
                    "scheme_name": scheme_label
                })

    st.session_state["schemes"] = pd.DataFrame(schemes)
    st.session_state["readings"] = pd.DataFrame(readings)
    st.session_state["jalmitras"] = jalmitras
    st.session_state["demo_generated"] = True
    st.success("✅ Demo data generated for ROKI RAY.")

# --------------------------- Metric computation ---------------------------
@st.cache_data
def compute_metrics(readings: pd.DataFrame, schemes: pd.DataFrame, so: str, start: str, end: str):
    r = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    s = ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name"])
    merged = r.merge(s[["id","scheme_name","functionality","so_name"]], left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))
    mask = (merged["functionality"] == "Functional") & (merged["so_name"] == so) & (merged["reading_date"] >= start) & (merged["reading_date"] <= end)
    last7 = merged.loc[mask].copy()
    if last7.empty:
        return last7, pd.DataFrame()
    last7["water_quantity"] = pd.to_numeric(last7["water_quantity"], errors="coerce").fillna(0.0).round(2)
    metrics = last7.groupby("jalmitra").agg(days_updated=("reading_date", lambda x: x.nunique()), total_water_m3=("water_quantity","sum")).reset_index()
    metrics["total_water_m3"] = metrics["total_water_m3"].round(2)
    return last7, metrics

# --------------------------- Demo Data Buttons ---------------------------
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns([2,1])
with col1:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=100, value=20)
    if st.button("Generate Demo Data"):
        generate_demo_data(int(total_schemes))
with col2:
    if st.button("Remove Demo Data"):
        reset_session_data(); st.warning("🗑️ All data removed.")
st.markdown("---")

# --------------------------- Header ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.header(f"{role} Dashboard — Placeholder"); st.stop()

so = "ROKI RAY"
today = datetime.date.today()
st.header(f"Section Officer Dashboard — {so}")
st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")

schemes = st.session_state["schemes"]
readings = st.session_state["readings"]
if schemes.empty:
    st.info("No schemes found. Generate demo data first."); st.stop()

# --------------------------- Overview ---------------------------
func_counts = schemes["functionality"].value_counts()
today_iso = today.isoformat()

merged_all = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]) \
             .merge(ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name"])[["id","scheme_name","functionality","so_name"]],
                    left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))

today_upd = merged_all[
    (merged_all["reading_date"] == today_iso) &
    (merged_all["functionality"] == "Functional") &
    (merged_all["so_name"] == so)
].copy()

def pick_display(row):
    r = row.get("scheme_name_reading", None)
    s = row.get("scheme_name_scheme", None)
    return r if pd.notna(r) and str(r).strip() != "" else s

if not today_upd.empty:
    today_upd["Scheme Display"] = today_upd.apply(pick_display, axis=1)

updated_count = int(today_upd["jalmitra"].nunique()) if not today_upd.empty else 0
total_functional = int(len(schemes[schemes["functionality"] == "Functional"]))
absent_count = max(total_functional - updated_count, 0)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values,
                  color=func_counts.index, color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    st.plotly_chart(fig1, use_container_width=True)
with col2:
    st.markdown("#### Jalmitra Updates (Today)")
    dfp = pd.DataFrame({"status":["Updated","Absent"],"count":[updated_count, absent_count]})
    fig2 = px.pie(dfp, names="status", values="count", color="status", color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --------------------------- Rankings ---------------------------
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7, metrics = compute_metrics(readings, schemes, so, start_date, end_date)

if not metrics.empty:
    metrics["score"] = 0.5 * (metrics["days_updated"]/7.0) + 0.5 * (metrics["total_water_m3"]/metrics["total_water_m3"].max())
    metrics = metrics.sort_values(by="score", ascending=False).reset_index(drop=True)
    metrics["Rank"] = metrics.index + 1
    rnd = random.Random(42)
    villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Tezpur","Sibsagar","Hajo","Nalbari","Moran"]
    metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]

    top_table = metrics.head(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","score"]]
    top_table.columns = ["Rank","Jalmitra","Scheme Name","Days Updated (last 7d)","Total Water (m³)","Score"]

    worst_table = metrics.tail(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","score"]]
    worst_table.columns = ["Rank","Jalmitra","Scheme Name","Days Updated (last 7d)","Total Water (m³)","Score"]

    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        st.dataframe(top_table.style.format({"Total Water (m³)":"{:.2f}","Score":"{:.3f}"})
                     .background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Greens"), height=360)
    with col_w:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.dataframe(worst_table.style.format({"Total Water (m³)":"{:.2f}","Score":"{:.3f}"})
                     .background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Reds_r"), height=360)

    st.markdown("**Tap a name below to view 7-day chart**")
    names = top_table["Jalmitra"].tolist() + worst_table["Jalmitra"].tolist()
    for name in names:
        if st.button(name, key=f"btn_{name}"):
            st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name

# --------------------------- Chart ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")
    st.subheader(f"7-Day Performance — {jm}")
    jm_data = last7[last7["jalmitra"] == jm] if not last7.empty else pd.DataFrame()
    if jm_data.empty:
        st.info("No readings for this Jalmitra in the last 7 days.")
    else:
        dates = [(today - datetime.timedelta(days=d)).isoformat() for d in reversed(range(7))]
        daily = jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates, fill_value=0).reset_index()
        daily["water_quantity"] = daily["water_quantity"].round(2)
        fig = px.bar(daily, x="reading_date", y="water_quantity",
                     labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                     title=f"{jm} — Daily Water Supplied (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"**Total (7 days):** {daily['water_quantity'].sum():.2f} m³  **Days Updated:** {(daily['water_quantity']>0).sum()}/7")
    if st.button("Close 7-day View"):
        st.session_state["selected_jalmitra"] = None

# --------------------------- BFM Readings Updated Today ---------------------------
st.markdown("---")
st.subheader("📅 BFM Readings Updated Today")
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    today_upd["Scheme Name"] = today_upd["Scheme Display"]
    today_upd["reading"] = pd.to_numeric(today_upd["reading"], errors="coerce").fillna(0).astype(int)
    today_upd["water_quantity"] = pd.to_numeric(today_upd["water_quantity"], errors="coerce").fillna(0.0).round(2)
    today_upd["BFM Reading Display"] = today_upd["reading"].apply(lambda x: f"{x:06d}")

    daily_bfm = today_upd[["jalmitra","Scheme Name","BFM Reading Display","reading_time","water_quantity"]].copy()
    daily_bfm.columns = ["Jalmitra","Scheme Name","BFM Reading","Reading Time","Water Quantity (m³)"]
    daily_bfm = daily_bfm.sort_values("Jalmitra").reset_index(drop=True)
    daily_bfm.insert(0, "S.No", range(1, len(daily_bfm)+1))

    sty = daily_bfm.style.format({"Water Quantity (m³)":"{:.2f}"})
    st.dataframe(sty.background_gradient(cmap="Blues", subset=["Water Quantity (m³)"]), height=360)

st.success(f"Dashboard ready for SO {so}. Demo data generated: {st.session_state['demo_generated']}")
