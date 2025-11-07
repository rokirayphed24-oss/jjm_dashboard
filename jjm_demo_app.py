# jjm_demo_app.py
# JJM Dashboard — Full app with period filter (7/15/30 days) and per-scheme ideal water
# - SO = ROKI RAY
# - Demo data generator (Assamese names), per-scheme ideal_per_day (20-100 m³)
# - compute_metrics supports arbitrary window length and returns ideal totals + quantity_score
# - Top/Worst tables include Ideal Water (m³) and use 50% days + 50% quantity_score
# - BFM Readings Updated Today table preserved
# - All UI & exports present, unique widget keys to avoid duplicate key errors

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Unified", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("For Section Officer **ROKI RAY** — Tap a name (buttons) to view performance.")
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
            elif c in ("water_quantity", "ideal_per_day"):
                df[c] = 0.0
            else:
                df[c] = ""
    return df

def init_state():
    st.session_state.setdefault("schemes", pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day"]))
    st.session_state.setdefault("readings", pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]))
    st.session_state.setdefault("jalmitras", [])
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)

init_state()

# --------------------------- Demo data functions ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    st.session_state["jalmitras"] = []
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False
    st.session_state["selected_jalmitra"] = None

def generate_demo_data(total_schemes:int=20, so_name:str="ROKI RAY"):
    """
    Generate demo:
    - Schemes with ideal_per_day (20-100 m³)
    - For Functional schemes, generate up to `FIXED_UPDATE_PROB` readings per day for the last 7 days
    - Jalmitra names from Assamese name list
    """
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

    # create schemes and assign an ideal_per_day (random 20-100 m³)
    for i in range(total_schemes):
        ideal_per_day = round(random.uniform(20.0, 100.0), 2)
        schemes.append({
            "id": i+1,
            "scheme_name": f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}",
            "functionality": random.choice(["Functional","Non-Functional"]),
            "so_name": so_name,
            "ideal_per_day": ideal_per_day
        })

    # create readings for functional schemes only; cap water_quantity to 100 and round to 2 decimals
    for i, s in enumerate(schemes):
        if s["functionality"] != "Functional":
            continue
        scheme_label = random.choice(villages) + " PWSS"
        jalmitra = jalmitras[i % len(jalmitras)]
        for d in range(7):  # create values for last 7 days
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                # morning times 6:00 AM - 11:45 AM
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

# --------------------------- Metrics computation ---------------------------
@st.cache_data
def compute_metrics(readings: pd.DataFrame, schemes: pd.DataFrame, so: str, start: str, end: str):
    """
    Returns:
      - lastN: merged readings (only functional schemes) in date window with scheme ideal_per_day attached
      - metrics: per-jalmitra metrics including:
          days_updated, total_water_m3, schemes_covered, ideal_total_Nd, quantity_score
    Note: ideal totals computed for the number of days in the window (end - start + 1).
    """
    r = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    s = ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name","ideal_per_day"])
    merged = r.merge(
        s[["id","scheme_name","functionality","so_name","ideal_per_day"]],
        left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme")
    )

    mask = (
        (merged["functionality"] == "Functional")
        & (merged["so_name"] == so)
        & (merged["reading_date"] >= start)
        & (merged["reading_date"] <= end)
    )
    lastN = merged.loc[mask].copy()
    if lastN.empty:
        return lastN, pd.DataFrame()

    # determine number of days in window inclusive
    try:
        days_count = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        if days_count <= 0:
            days_count = 1
    except Exception:
        days_count = 7

    # ensure numeric
    lastN["water_quantity"] = pd.to_numeric(lastN["water_quantity"], errors="coerce").fillna(0.0).round(2)
    lastN["ideal_per_day"] = pd.to_numeric(lastN.get("ideal_per_day", 0.0), errors="coerce").fillna(0.0)

    agg = lastN.groupby("jalmitra").agg(
        days_updated=("reading_date", lambda x: x.nunique()),
        total_water_m3=("water_quantity", "sum"),
        schemes_covered=("scheme_id", lambda x: x.nunique())
    ).reset_index()

    scheme_ideal = lastN[["jalmitra","scheme_id","ideal_per_day"]].drop_duplicates(subset=["jalmitra","scheme_id"])
    scheme_ideal["ideal_Nd"] = scheme_ideal["ideal_per_day"] * float(days_count)
    ideal_sum = scheme_ideal.groupby("jalmitra")["ideal_Nd"].sum().reset_index().rename(columns={"ideal_Nd":"ideal_total_Nd"})

    metrics = agg.merge(ideal_sum, on="jalmitra", how="left")
    metrics["ideal_total_Nd"] = metrics["ideal_total_Nd"].fillna(0.0).round(2)

    # quantity score fraction of ideal achieved (0..1)
    def compute_qs(row):
        ideal = row["ideal_total_Nd"]
        water = row["total_water_m3"]
        if ideal <= 0:
            return 0.0
        return min(float(water) / float(ideal), 1.0)

    metrics["quantity_score"] = metrics.apply(compute_qs, axis=1)
    metrics["days_updated"] = metrics["days_updated"].astype(int)
    metrics["total_water_m3"] = metrics["total_water_m3"].astype(float).round(2)
    metrics["quantity_score"] = metrics["quantity_score"].astype(float).round(3)

    # attach days_count as attribute for caller convenience
    metrics.attrs["days_count"] = days_count

    return lastN, metrics

# --------------------------- Demo data UI ---------------------------
st.markdown("### 🧪 Demo Data Management")
col1, col2 = st.columns([2,1])
with col1:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20)
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
    st.info("No schemes found. Generate demo data first.")
    st.stop()

# --------------------------- Overview pies ---------------------------
func_counts = schemes["functionality"].value_counts()
today_iso = today.isoformat()

merged_all = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]) \
             .merge(ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name","ideal_per_day"])[["id","scheme_name","functionality","so_name","ideal_per_day"]],
                    left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))

today_upd = merged_all[
    (merged_all["reading_date"] == today_iso) &
    (merged_all["functionality"] == "Functional") &
    (merged_all["so_name"] == so)
].copy()

def pick_display(row):
    r = row.get("scheme_name_reading", None)
    s = row.get("scheme_name_scheme", None)
    if pd.notna(r) and str(r).strip() != "":
        return r
    if pd.notna(s) and str(s).strip() != "":
        return s
    return ""

if not today_upd.empty:
    if "scheme_name_reading" not in today_upd.columns and "scheme_name" in today_upd.columns:
        today_upd["scheme_name_reading"] = today_upd["scheme_name"]
    if "scheme_name_scheme" not in today_upd.columns and "scheme_name" in today_upd.columns:
        today_upd["scheme_name_scheme"] = today_upd["scheme_name"]
    today_upd["Scheme Display"] = today_upd.apply(pick_display, axis=1)

updated_count = int(today_upd["jalmitra"].nunique()) if not today_upd.empty else 0
total_functional = int(len(schemes[schemes["functionality"] == "Functional"]))
absent_count = max(total_functional - updated_count, 0)

# Pies
if view_mode == "Web View":
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Scheme Functionality")
        fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                      color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
        fig1.update_traces(textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True, height=220)
    with col2:
        st.markdown("#### Jalmitra Updates (Today)")
        df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[updated_count, absent_count]})
        if df_part["count"].sum() == 0:
            df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[0, total_functional if total_functional>0 else 1]})
        fig2 = px.pie(df_part, names='status', values='count', color='status',
                      color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True, height=220)
else:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                  color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True, height=240)

    st.markdown("#### Jalmitra Updates (Today)")
    df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[updated_count, absent_count]})
    if df_part["count"].sum() == 0:
        df_part = pd.DataFrame({"status":["Updated","Absent"], "count":[0, total_functional if total_functional>0 else 1]})
    fig2 = px.pie(df_part, names='status', values='count', color='status',
                  color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True, height=240)

st.markdown("---")

# --------------------------- Rankings ---------------------------
st.subheader("🏅 Jalmitra Performance — Top & Worst")
# Period selector
period = st.selectbox("Show performance for", [7, 15, 30], index=0, format_func=lambda x: f"{x} days")
start_date = (today - datetime.timedelta(days=period-1)).isoformat()
end_date = today_iso

lastN, metrics = compute_metrics(readings, schemes, so, start_date, end_date)

if lastN.empty or metrics.empty:
    st.info(f"No readings in the last {period} days.")
else:
    metrics["days_norm"] = metrics["days_updated"] / float(period)
    metrics["score"] = (0.5 * metrics["days_norm"]) + (0.5 * metrics["quantity_score"])
    metrics = metrics.sort_values(by=["score","total_water_m3"], ascending=False).reset_index(drop=True)
    metrics["Rank"] = metrics.index + 1

    # deterministic scheme names for display
    villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    rnd = random.Random(42)
    metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]

    metrics["ideal_total_Nd"] = metrics.get("ideal_total_Nd", 0.0).round(2)

    top_table = metrics.head(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
    top_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

    worst_table = metrics.sort_values(by='score', ascending=True).head(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
    worst_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

    # styled tables
    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown(f"### 🟢 Top 10 Performing Jalmitras — last {period} days")
        st.dataframe(top_table.style.format({"Total Water (m³)":"{:.2f}","Ideal Water (m³)":"{:.2f}","Score":"{:.3f}"}).background_gradient(subset=[f"Days Updated (last {period}d)","Total Water (m³)","Score"], cmap="Greens"), height=360)
        st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode("utf-8"), "top_10_jalmitras.csv")
    with col_w:
        st.markdown(f"### 🔴 Worst 10 Performing Jalmitras — last {period} days")
        st.dataframe(worst_table.style.format({"Total Water (m³)":"{:.2f}","Ideal Water (m³)":"{:.2f}","Score":"{:.3f}"}).background_gradient(subset=[f"Days Updated (last {period}d)","Total Water (m³)","Score"], cmap="Reds_r"), height=360)
        st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode("utf-8"), "worst_10_jalmitras.csv")

    # clickable names below tables (unique keys include period)
    st.markdown("**Tap a name below to open the performance chart**")
    top_names = top_table["Jalmitra"].tolist()
    worst_names = worst_table["Jalmitra"].tolist()

    if view_mode == "Web View":
        with st.container():
            st.markdown("**Top 10 — Tap name**")
            if top_names:
                cols = st.columns(len(top_names))
                for i, name in enumerate(top_names):
                    if cols[i].button(name, key=f"btn_top_{period}_{i}_{name}"):
                        st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name

        with st.container():
            st.markdown("**Worst 10 — Tap name**")
            if worst_names:
                cols = st.columns(len(worst_names))
                for i, name in enumerate(worst_names):
                    if cols[i].button(name, key=f"btn_worst_{period}_{i}_{name}"):
                        st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name
    else:
        st.markdown("**Top 10 — Tap a name**")
        for i, name in enumerate(top_names):
            if st.button(name, key=f"pbtn_top_{period}_{i}_{name}"):
                st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name
        st.markdown("**Worst 10 — Tap a name**")
        for i, name in enumerate(worst_names):
            if st.button(name, key=f"pbtn_worst_{period}_{i}_{name}"):
                st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name

# --------------------------- Show performance chart when name selected ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")
    st.subheader(f"Performance — {jm}")
    # recompute for selected window (use the same period)
    last_window, _ = compute_metrics(readings, schemes, so, start_date, end_date)
    jm_data = last_window[last_window["jalmitra"] == jm] if (not last_window.empty) else pd.DataFrame()
    if jm_data.empty:
        st.info("No readings for this Jalmitra in the selected window.")
    else:
        dates = [(datetime.date.today() - datetime.timedelta(days=d)).isoformat() for d in reversed(range(period))]
        daily = jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates, fill_value=0).reset_index()
        daily["water_quantity"] = daily["water_quantity"].round(2)
        fig = px.bar(daily, x="reading_date", y="water_quantity",
                     labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                     title=f"{jm} — Daily Water Supplied (Last {period} Days)")
        st.plotly_chart(fig, use_container_width=True, height=380)
        st.markdown(f"**Total ({period} days):** {daily['water_quantity'].sum():.2f} m³  **Days Updated:** {(daily['water_quantity']>0).sum()}/{period}")
    # close button
    if view_mode == "Web View":
        if st.button("Close View"):
            st.session_state["selected_jalmitra"] = None
    else:
        if st.button("Close View (Phone)"):
            st.session_state["selected_jalmitra"] = None

st.markdown("---")

# --------------------------- BFM Readings Updated Today ---------------------------
st.subheader("📅 BFM Readings Updated Today")
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    today_upd["Scheme Name"] = today_upd.get("Scheme Display", "")
    today_upd["reading"] = pd.to_numeric(today_upd.get("reading", 0), errors="coerce").fillna(0).astype(int)
    today_upd["water_quantity"] = pd.to_numeric(today_upd.get("water_quantity", 0.0), errors="coerce").fillna(0.0).round(2)
    today_upd["BFM Reading Display"] = today_upd["reading"].apply(lambda x: f"{x:06d}")

    # Build table with Reading Time immediately after Reading
    daily_bfm = today_upd[["jalmitra","Scheme Display","BFM Reading Display","reading_time","water_quantity"]].copy()
    daily_bfm.columns = ["Jalmitra","Scheme Name","BFM Reading","Reading Time","Water Quantity (m³)"]
    daily_bfm = daily_bfm.sort_values("Jalmitra").reset_index(drop=True)
    daily_bfm.insert(0, "S.No", range(1, len(daily_bfm)+1))

    # Display with styling
    try:
        sty = daily_bfm.style.format({"Water Quantity (m³)":"{:.2f}"})
        st.dataframe(sty.background_gradient(cmap="Blues", subset=["Water Quantity (m³)"]), height=360)
    except Exception:
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
