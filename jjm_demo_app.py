# jjm_demo_app.py
# JJM Dashboard — Full (fixed Pandas Styler BFM-format error)
# - Styled Top/Worst tables
# - Clickable "View" per Jalmitra (7-day chart)
# - "📅 BFM Readings Updated Today" with Scheme Name and safe formatting

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px
from typing import Tuple

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Full", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("For Section Officer **ROKI RAY** — Jalmitra performance, daily updates, and readings.")
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
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)

init_state()

# --------------------------- Demo data generator ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    st.session_state["jalmitras"] = []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
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

    # create schemes
    for i in range(total_schemes):
        schemes.append({
            "id": i+1,
            "scheme_name": f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}",
            "functionality": random.choice(["Functional","Non-Functional"]),
            "so_name": so_name
        })

    # create readings for functional schemes only; include scheme_name label per reading
    for i, s in enumerate(schemes):
        if s["functionality"] != "Functional":
            continue
        scheme_label = random.choice(villages) + " PWSS"
        jalmitra = jalmitras[i % len(jalmitras)]
        for d in range(7):
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                readings.append({
                    "id": len(readings) + 1,
                    "scheme_id": s["id"],
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date_iso,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity": round(random.uniform(40.0,350.0),2),
                    "scheme_name": scheme_label  # store label in reading
                })

    st.session_state["schemes"] = pd.DataFrame(schemes)
    st.session_state["readings"] = pd.DataFrame(readings)
    st.session_state["jalmitras"] = jalmitras
    st.session_state["demo_generated"] = True
    st.success("✅ Demo data generated for ROKI RAY.")

# --------------------------- Compute metrics ---------------------------
@st.cache_data
def compute_metrics(readings, schemes, so, start, end):
    r = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    s = ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name"])
    # merge; keep both reading's scheme_name and scheme table's scheme_name using suffixes
    m = r.merge(s[["id","scheme_name","functionality","so_name"]], left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))
    mask = (m["functionality"] == "Functional") & (m["so_name"] == so) & (m["reading_date"] >= start) & (m["reading_date"] <= end)
    last7 = m.loc[mask].copy()
    if last7.empty:
        return last7, pd.DataFrame()
    metrics = (last7.groupby("jalmitra")
               .agg(days_updated=("reading_date", lambda x: x.nunique()), total_water_m3=("water_quantity", "sum"))
               .reset_index())
    return last7, metrics

# --------------------------- Demo data UI ---------------------------
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns([2,1])
with col1:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=100, value=20)
    if st.button("Generate Demo Data"):
        generate_demo_data(int(total_schemes))
with col2:
    if st.button("Remove Demo Data"):
        reset_session_data()
        st.warning("🗑️ All data removed.")
st.markdown("---")

# --------------------------- Dashboard ---------------------------
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"])
if role != "Section Officer":
    st.header(f"{role} Dashboard — Placeholder")
    st.stop()

so = "ROKI RAY"
today = datetime.date.today()
st.header(f"Section Officer Dashboard — {so}")
st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")

schemes = st.session_state["schemes"]
readings = st.session_state["readings"]
if schemes.empty:
    st.info("No schemes found. Generate demo data first.")
    st.stop()

# --------------------------- Pies ---------------------------
func_counts = schemes["functionality"].value_counts()
today_iso = today.isoformat()

# merge readings and schemes safely (we will pick scheme display from reading if present)
merged_all = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]) \
             .merge(ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name"])[["id","scheme_name","functionality","so_name"]],
                    left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))

# Build today's updates subset
today_upd = merged_all[
    (merged_all["reading_date"] == today_iso) &
    (merged_all["functionality"] == "Functional") &
    (merged_all["so_name"] == so)
].copy()

# create a display scheme name column: prefer reading's scheme_name if non-empty, else fallback to scheme table name
def pick_scheme_display(row):
    val_reading = row.get("scheme_name_reading", None)
    val_scheme = row.get("scheme_name_scheme", None)
    # prefer reading-level value
    if pd.notna(val_reading) and str(val_reading).strip() != "":
        return val_reading
    if pd.notna(val_scheme) and str(val_scheme).strip() != "":
        return val_scheme
    return ""

if not today_upd.empty:
    # If merge produced different naming, normalize columns
    if "scheme_name_reading" not in today_upd.columns and "scheme_name" in today_upd.columns:
        today_upd["scheme_name_reading"] = today_upd["scheme_name"]
    if "scheme_name_scheme" not in today_upd.columns and "scheme_name" in today_upd.columns:
        today_upd["scheme_name_scheme"] = today_upd["scheme_name"]
    today_upd["Scheme Display"] = today_upd.apply(pick_scheme_display, axis=1)

# compute updated/absent counts for pie
updated_count = int(today_upd["jalmitra"].nunique()) if not today_upd.empty else 0
total_functional = int(len(schemes[schemes["functionality"] == "Functional"]))
absent_count = max(total_functional - updated_count, 0)

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                  color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True)

with col2:
    st.markdown("#### Jalmitra Updates (Today)")
    df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[updated_count, absent_count]})
    if df_part["count"].sum() == 0:
        df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[0, total_functional if total_functional>0 else 1]})
    fig2 = px.pie(df_part, names="status", values="count", color="status", color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --------------------------- Top/Worst + View ---------------------------
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7, metrics = compute_metrics(readings, schemes, so, start_date, end_date)

if last7.empty:
    st.info("No readings in last 7 days.")
else:
    # compute score and ranking
    metrics["score"] = 0.5 * (metrics["days_updated"] / 7.0) + 0.5 * (metrics["total_water_m3"] / metrics["total_water_m3"].max())
    metrics = metrics.sort_values(by=["score","total_water_m3"], ascending=False).reset_index(drop=True)
    metrics["Rank"] = metrics.index + 1

    # assign deterministic scheme names (display) for metrics table (illustrative)
    villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    rnd = random.Random(42)
    metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]

    top_n = 10
    top_table = metrics.head(top_n)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","score"]].copy()
    top_table.columns = ["Rank","Jalmitra","Scheme Name","Days Updated (last 7d)","Total Water (m³)","Score"]

    worst_table = metrics.sort_values(by="score", ascending=True).head(top_n)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","score"]].copy()
    worst_table.columns = ["Rank","Jalmitra","Scheme Name","Days Updated (last 7d)","Total Water (m³)","Score"]

    # show styled tables
    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown("### 🟢 Top 10 Performing Jalmitras")
        st.dataframe(top_table.style.format({"Total Water (m³)":"{:.2f}", "Score":"{:.3f}"}).background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Greens"), height=300)
    with col_w:
        st.markdown("### 🔴 Worst 10 Performing Jalmitras")
        st.dataframe(worst_table.style.format({"Total Water (m³)":"{:.2f}", "Score":"{:.3f}"}).background_gradient(subset=["Days Updated (last 7d)","Total Water (m³)","Score"], cmap="Reds_r"), height=300)

    # compact rows with View buttons (preserve styled tables but allow clicks)
    st.markdown("**Tap View to see 7-day chart**")
    col_t2, col_w2 = st.columns([1,1])
    with col_t2:
        st.markdown("Top 10 — Actions")
        for i, row in top_table.reset_index(drop=True).iterrows():
            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.4,2.2,1.2,1.2,0.9,0.9])
            c0.write(row["Rank"])
            c1.write(row["Jalmitra"])
            c2.write(row["Scheme Name"])
            c3.write(row["Days Updated (last 7d)"])
            c4.write(f"{row['Total Water (m³)']:,}")
            c5.write(f"{row['Score']:.3f}")
            btn_key = f"view_top_{i}_{row['Jalmitra']}"
            if c6.button("View", key=btn_key):
                st.session_state["selected_jalmitra"] = row["Jalmitra"]
        st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode("utf-8"), "top_10_jalmitras.csv")

    with col_w2:
        st.markdown("Worst 10 — Actions")
        for i, row in worst_table.reset_index(drop=True).iterrows():
            c0, c1, c2, c3, c4, c5, c6 = st.columns([0.6,1.4,2.2,1.2,1.2,0.9,0.9])
            c0.write(row["Rank"])
            c1.write(row["Jalmitra"])
            c2.write(row["Scheme Name"])
            c3.write(row["Days Updated (last 7d)"])
            c4.write(f"{row['Total Water (m³)']:,}")
            c5.write(f"{row['Score']:.3f}")
            btn_key = f"view_worst_{i}_{row['Jalmitra']}"
            if c6.button("View", key=btn_key):
                st.session_state["selected_jalmitra"] = row["Jalmitra"]
        st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode("utf-8"), "worst_10_jalmitras.csv")

# --------------------------- View chart ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")
    st.subheader(f"7-day Performance — {jm}")
    # recompute/ensure last7 available
    if 'last7' not in locals() or last7 is None:
        last7, _ = compute_metrics(readings, schemes, so, start_date, end_date)
    jm_data = last7[last7["jalmitra"] == jm] if (not last7.empty) else pd.DataFrame()
    if jm_data.empty:
        st.info("No readings for this Jalmitra.")
    else:
        dates = [(today - datetime.timedelta(days=d)).isoformat() for d in reversed(range(7))]
        daily = jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates, fill_value=0).reset_index()
        fig = px.bar(daily, x="reading_date", y="water_quantity",
                     labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                     title=f"{jm} — Daily Water Supplied (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True, height=380)
        st.markdown(f"**Total:** {daily['water_quantity'].sum():.2f} m³  **Days Updated:** {(daily['water_quantity']>0).sum()}/7")
    if st.button("Close View"):
        st.session_state["selected_jalmitra"] = None

# --------------------------- NEW — Daily BFM Readings ---------------------------
st.markdown("---")
st.subheader("📅 BFM Readings Updated Today")

# If there are readings today (today_upd), pick the most recent reading per jalmitra by reading_time and show the scheme_display we computed
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    # Normalize column names and ensure values exist
    cols_to_ensure = ["jalmitra", "reading", "water_quantity", "reading_time", "scheme_name_reading", "scheme_name_scheme", "Scheme Display"]
    today_upd = ensure_columns(today_upd.copy(), cols_to_ensure)

    # Normalize reading_time for sorting - if missing, set to "00:00:00"
    today_upd["reading_time_norm"] = today_upd["reading_time"].astype(str).replace("", "00:00:00")

    # Sort by jalmitra then reading_time descending so the first per jalmitra is the latest
    today_upd_sorted = today_upd.sort_values(by=["jalmitra", "reading_time_norm"], ascending=[True, False])
    latest_per_jm = today_upd_sorted.drop_duplicates(subset=["jalmitra"], keep="first").copy()

    # Resolve scheme name to display: prefer reading-level scheme_name, then merged scheme name
    def scheme_for_row(row):
        sd = row.get("Scheme Display", "")
        if pd.notna(sd) and str(sd).strip() != "":
            return sd
        s_read = row.get("scheme_name_reading", "")
        if pd.notna(s_read) and str(s_read).strip() != "":
            return s_read
        s_scheme = row.get("scheme_name_scheme", "")
        if pd.notna(s_scheme) and str(s_scheme).strip() != "":
            return s_scheme
        return ""

    latest_per_jm["Scheme Name"] = latest_per_jm.apply(scheme_for_row, axis=1)

    # Ensure reading numeric and water qty numeric; fill NaNs
    latest_per_jm["reading"] = pd.to_numeric(latest_per_jm["reading"], errors="coerce").fillna(0).astype(int)
    latest_per_jm["water_quantity"] = pd.to_numeric(latest_per_jm["water_quantity"], errors="coerce").fillna(0.0)

    # Create display-safe formatted BFM Reading string (zero-padded) to avoid Styler integer formatting issues
    latest_per_jm["BFM Reading Display"] = latest_per_jm["reading"].apply(lambda x: f"{int(x):06d}")

    # Build the daily table for display
    daily_bfm = latest_per_jm[["jalmitra", "Scheme Name", "BFM Reading Display", "water_quantity"]].copy()
    daily_bfm.columns = ["Jalmitra", "Scheme Name", "BFM Reading", "Water Quantity (m³)"]
    daily_bfm = daily_bfm.sort_values("Jalmitra").reset_index(drop=True)

    # Display: since BFM Reading is already a string formatted, format only Water Quantity
    sty = daily_bfm.style.format({"Water Quantity (m³)":"{:.2f}"})
    # apply blue gradient only to the numeric column; Styler expects numeric col for gradient, so create a helper column
    # we'll temporarily add a numeric copy for gradient and then drop it from display if necessary.
    # But pandas Styler allows subset selection; gradient will skip non-numeric cells.
    try:
        st.dataframe(sty.background_gradient(cmap="Blues", subset=["Water Quantity (m³)"]), height=360)
    except Exception:
        # last-resort: render without styling if styler fails
        st.dataframe(daily_bfm, height=360)

# --------------------------- Export ---------------------------
st.markdown("---")
st.subheader("📤 Export Snapshot")
st.download_button("Schemes CSV", schemes.to_csv(index=False).encode("utf-8"), "schemes.csv")
st.download_button("Readings CSV", readings.to_csv(index=False).encode("utf-8"), "readings.csv")
try:
    st.download_button("Metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "metrics.csv")
except Exception:
    st.info("Metrics CSV not available.")

st.success(f"Dashboard ready for SO {so}. Demo data generated: {st.session_state['demo_generated']}")
