# jjm_demo_app.py
# JJM Dashboard — Unified with improved ideal-line chart & full-window CSV export
# - Export includes all days in selected window (zeros for no-reading days)
# - Chart bars green when actual >= ideal, red when actual < ideal
# - Ideal line drawn across full date range
# - Rest of app unchanged

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Unified", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("---")

# --------------------------- Helpers & session init ---------------------------
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
    st.session_state.setdefault("schemes", pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"]))
    st.session_state.setdefault("readings", pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"
    ]))
    st.session_state.setdefault("jalmitras", [])
    st.session_state.setdefault("jalmitras_map", {})  # for multi-SO demo: so_name -> list of jalmitras
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)
    st.session_state.setdefault("selected_so_from_aee", None)
    st.session_state.setdefault("role", "Section Officer")
    st.session_state.setdefault("view_mode", "Web View")

init_state()

# --------------------------- Core data functions ---------------------------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"])
    st.session_state["jalmitras"] = []
    st.session_state["jalmitras_map"] = {}
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False
    st.session_state["selected_jalmitra"] = None
    st.session_state["selected_so_from_aee"] = None

def generate_demo_data(total_schemes:int=20, so_name:str="ROKI RAY"):
    FIXED_UPDATE_PROB = 0.85
    assamese = [
        "Biren","Nagen","Rahul","Vikram","Debojit","Anup","Kamal","Ranjit","Himangshu",
        "Pranjal","Rupam","Dilip","Utpal","Amit","Jayanta","Hemanta","Rituraj","Dipankar",
        "Bikash","Dhruba","Subham","Pritam","Saurav","Bijoy","Manoj"
    ]
    jalmitras = (assamese * 5)[:max(total_schemes, len(assamese))]
    random.shuffle(jalmitras)
    jalmitras = jalmitras[:total_schemes]
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
        scheme_label = random.choice(villages) + " PWSS"
        schemes.append({
            "id": i+1,
            "scheme_name": f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}",
            "functionality": random.choice(["Functional","Non-Functional"]),
            "so_name": so_name,
            "ideal_per_day": ideal_per_day,
            "scheme_label": scheme_label
        })

    # create readings for functional schemes only; create for last 30 days
    days_to_generate = 30
    rid = 1
    for i, s in enumerate(schemes):
        if s["functionality"] != "Functional":
            continue
        scheme_label = s["scheme_label"]
        jalmitra = jalmitras[i % len(jalmitras)]
        for d in range(days_to_generate):
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                hour = random.randint(6, 11)
                minute = random.choice([0, 15, 30, 45])
                ampm = "AM"
                time_str = f"{hour}:{minute:02d} {ampm}"
                water_qty = round(random.uniform(10.0, 100.0), 2)
                readings.append({
                    "id": rid,
                    "scheme_id": s["id"],
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date_iso,
                    "reading_time": time_str,
                    "water_quantity": water_qty,
                    "scheme_name": scheme_label,
                    "so_name": so_name
                })
                rid += 1

    st.session_state["schemes"] = pd.DataFrame(schemes)
    st.session_state["readings"] = pd.DataFrame(readings)
    st.session_state["jalmitras"] = jalmitras
    st.session_state["jalmitras_map"] = {so_name: jalmitras}
    st.session_state["demo_generated"] = True
    st.success(f"✅ Demo data generated for {so_name}.")

# Multi-SO generator for AEE
def generate_multi_so_demo(num_sos=14, schemes_per_so=18, max_days=30):
    random.seed(42)
    so_names = [
        "ROKI RAY", "Sanjay Das", "Anup Bora", "Ranjit Kalita", "Bikash Deka", "Manoj Das",
        "Dipankar Nath", "Himangshu Deka", "Kamal Choudhury", "Rituraj Das", "Debojit Gogoi",
        "Utpal Saikia", "Pritam Bora", "Amit Baruah"
    ][:num_sos]

    schemes_rows = []
    readings_rows = []
    jalmitras_map = {}
    sid = 1
    rid = 1
    today = datetime.date.today()
    villages = [
        "Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar",
        "Jorhat","Hajo","Tihu","Kokrajhar","Nalbari","Barpeta","Rangia","Goalpara","Dhemaji",
        "Dibrugarh","Mariani","Sonari"
    ]
    assamese = [
        "Biren","Nagen","Rahul","Vikram","Debojit","Anup","Kamal","Ranjit","Himangshu",
        "Pranjal","Rupam","Dilip","Utpal","Amit","Jayanta","Hemanta","Rituraj","Dipankar",
        "Bikash","Dhruba","Subham","Pritam","Saurav","Bijoy","Manoj"
    ]

    for so in so_names:
        rng = random.Random(abs(hash(so)) % (2**32))
        n_jm = rng.randint(6, 12)
        jm_list = [rng.choice(assamese) + f"_{i+1}" for i in range(n_jm)]
        jalmitras_map[so] = jm_list

        for i in range(schemes_per_so):
            ideal_per_day = round(rng.uniform(20.0, 100.0), 2)
            scheme_label = rng.choice(villages) + " PWSS"
            func = "Functional" if rng.random() > 0.25 else "Non-Functional"
            schemes_rows.append({
                "id": sid,
                "scheme_name": f"Scheme_{sid}_{so.split()[0]}",
                "functionality": func,
                "so_name": so,
                "ideal_per_day": ideal_per_day,
                "scheme_label": scheme_label
            })
            sid += 1

        for jm in jm_list:
            jm_rng = random.Random(abs(hash(so + jm)) % (2**32))
            so_func_ids = [r["id"] for r in schemes_rows if r["so_name"] == so and r["functionality"] == "Functional"]
            if not so_func_ids:
                so_func_ids = [r["id"] for r in schemes_rows if r["so_name"] == so]
            for d in range(max_days):
                date_iso = (today - datetime.timedelta(days=d)).isoformat()
                if jm_rng.random() < 0.78:
                    hour = jm_rng.randint(6, 11)
                    minute = jm_rng.choice([0,15,30,45])
                    time_str = f"{hour}:{minute:02d} AM"
                    water_qty = round(jm_rng.uniform(10.0, 100.0), 2)
                    readings_rows.append({
                        "id": rid,
                        "scheme_id": jm_rng.choice(so_func_ids),
                        "jalmitra": jm,
                        "reading": jm_rng.choice([110010,215870,150340,189420,200015,234870]),
                        "reading_date": date_iso,
                        "reading_time": time_str,
                        "water_quantity": water_qty,
                        "so_name": so
                    })
                    rid += 1

    schemes_df = pd.DataFrame(schemes_rows)
    readings_df = pd.DataFrame(readings_rows)
    if not readings_df.empty and not schemes_df.empty:
        readings_df = readings_df.merge(schemes_df[["id","scheme_name"]], left_on="scheme_id", right_on="id", how="left", suffixes=("","_scheme"))
        if "scheme_name_scheme" in readings_df.columns:
            readings_df["scheme_name"] = readings_df["scheme_name_scheme"].combine_first(readings_df["scheme_name"])
            readings_df.drop(columns=[c for c in readings_df.columns if c.endswith("_scheme")], inplace=True)

    st.session_state["schemes"] = schemes_df
    st.session_state["readings"] = readings_df
    st.session_state["jalmitras_map"] = jalmitras_map
    st.session_state["next_scheme_id"] = sid
    st.session_state["next_reading_id"] = rid
    st.session_state["demo_generated"] = True
    st.success("✅ Multi-SO demo data generated for AEE (14 SOs).")

# --------------------------- compute_metrics ---------------------------
@st.cache_data
def compute_metrics(readings: pd.DataFrame, schemes: pd.DataFrame, so: str, start: str, end: str):
    r = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"])
    s = ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])

    merged = r.merge(
        s[["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"]],
        left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme")
    )

    if "so_name_reading" in merged.columns or "so_name_scheme" in merged.columns:
        merged["so_name"] = merged.get("so_name_reading").combine_first(merged.get("so_name_scheme"))
    else:
        merged["so_name"] = merged.get("so_name").fillna("")

    if "scheme_name_reading" in merged.columns or "scheme_name_scheme" in merged.columns:
        merged["Scheme Display"] = merged.get("scheme_name_reading").combine_first(merged.get("scheme_name_scheme"))
    else:
        merged["Scheme Display"] = merged.get("scheme_name").fillna("")

    mask = (
        (merged.get("functionality", "") == "Functional")
        & (merged.get("so_name", "") == so)
        & (merged.get("reading_date", "") >= start)
        & (merged.get("reading_date", "") <= end)
    )
    lastN = merged.loc[mask].copy()
    if lastN.empty:
        return lastN, pd.DataFrame()

    try:
        days_count = (pd.to_datetime(end) - pd.to_datetime(start)).days + 1
        if days_count <= 0:
            days_count = 1
    except Exception:
        days_count = 7

    lastN["water_quantity"] = pd.to_numeric(lastN.get("water_quantity", 0.0), errors="coerce").fillna(0.0).round(2)
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
    metrics.attrs["days_count"] = days_count
    return lastN, metrics

# --------------------------- Sidebar: AEE demo controls (kept) ---------------------------
st.sidebar.header("Demo Controls")
if st.sidebar.button("Generate multi-SO demo (14 SOs)"):
    generate_multi_so_demo(num_sos=14, schemes_per_so=18, max_days=30)
if st.sidebar.button("Clear demo data (sidebar)"):
    reset_session_data()
    st.sidebar.warning("Session demo data cleared (sidebar).")
st.sidebar.markdown("---")
st.sidebar.write("Use Section Officer view (role selector) to run single-SO demo generator.")

# --------------------------- Role & view selection ---------------------------
st.markdown("### View Mode & Role")
selected_view = st.radio("View Mode", ["Web View", "Phone View"], horizontal=True, key="view_mode")
role = st.selectbox("Select Role", ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"], key="role")
st.markdown("---")

# --------------------------- Section Officer: single-SO demo controls (only when role==Section Officer) ---------------------------
if role == "Section Officer":
    st.markdown("### 🧪 Demo Data Management (Section Officer)")
    col1, col2 = st.columns([2,1])
    with col1:
        total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20, key="so_total_schemes")
        if st.button("Generate Demo Data"):
            generate_demo_data(int(total_schemes))
    with col2:
        if st.button("Remove Demo Data"):
            reset_session_data(); st.warning("🗑️ All data removed.")
    st.markdown("---")

# --------------------------- AEE page (only shown when role == Assistant Executive Engineer) ---------------------------
if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer Dashboard (Aggregated from SOs)")
    st.markdown(f"**AEE:** Er. ROKI RAY • **Subdivision:** Guwahati")
    st.markdown(f"**DATE:** {datetime.date.today().strftime('%A, %d %B %Y').upper()}")
    st.markdown("---")

    # In-page AEE controls
    st.markdown("#### AEE demo controls (generate or remove SOs under this AEE)")
    ac1, ac2, ac3 = st.columns([2,1,1])
    with ac1:
        aee_num_sos = st.number_input("Number of SOs to generate", min_value=1, max_value=30, value=14, key="aee_num_sos")
        aee_schemes_per_so = st.number_input("Schemes per SO", min_value=4, max_value=50, value=18, key="aee_schemes_per_so")
    with ac2:
        if st.button("Generate AEE demo (in-page)"):
            generate_multi_so_demo(num_sos=int(aee_num_sos), schemes_per_so=int(aee_schemes_per_so), max_days=30)
    with ac3:
        if st.button("Remove AEE demo (in-page)"):
            st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])
            st.session_state["readings"] = pd.DataFrame(columns=[
                "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"])
            st.session_state["jalmitras_map"] = {}
            st.session_state["demo_generated"] = False
            st.success("✅ AEE demo removed from session (in-page).")

    st.markdown("---")
    if not st.session_state["demo_generated"]:
        st.info("For AEE view, generate multi-SO demo data using the buttons above (in-page) or sidebar.")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Scheme Functionality (all SOs)")
        if not st.session_state["schemes"].empty:
            func_counts = st.session_state["schemes"]["functionality"].value_counts()
            fig = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                         color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True, height=260)
        else:
            st.write("No schemes available.")
    with col2:
        st.subheader("SO Updates (today)")
        today_iso = datetime.date.today().isoformat()
        if not st.session_state["readings"].empty:
            today_updates = st.session_state["readings"][st.session_state["readings"]["reading_date"] == today_iso]
            upd_counts = today_updates.groupby("so_name")["jalmitra"].nunique().fillna(0).astype(int)
            total_updates = int(upd_counts.sum())
            total_functional = int(len(st.session_state["schemes"][st.session_state["schemes"]["functionality"] == "Functional"]))
            df_upd = pd.DataFrame({"status":["Updated today (unique jalmitras)","Other (approx)"], "count":[total_updates, max(total_functional - total_updates, 0)]})
            fig2 = px.pie(df_upd, names="status", values="count", color="status",
                          color_discrete_map={"Updated today (unique jalmitras)":"#4CAF50","Other (approx)":"#F44336"})
            fig2.update_traces(textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True, height=260)
        else:
            st.write("No readings available.")

    st.markdown("---")
    st.subheader("Section Officer performance (aggregated from Jalmitra scores)")
    period = st.selectbox("Select window (days)", [7,15,30], index=0, key="aee_period")
    st.markdown(f"Showing performance for last **{period} days**")

    # compute AEE metrics
    def compute_jalmitra_metrics_for_period(readings_df, schemes_df, period_days):
        start_date = (datetime.date.today() - datetime.timedelta(days=period_days-1)).isoformat()
        end_date = datetime.date.today().isoformat()
        sel = readings_df[(readings_df["reading_date"] >= start_date) & (readings_df["reading_date"] <= end_date)].copy()
        if sel.empty:
            return pd.DataFrame(), pd.DataFrame()
        grouped = sel.groupby(["so_name","jalmitra"]).agg(
            days_updated = ("reading_date", lambda x: x.nunique()),
            total_water = ("water_quantity", "sum")
        ).reset_index()

        # baseline list of jalmitras under each SO
        rows = []
        for so, jlist in st.session_state["jalmitras_map"].items():
            for jm in jlist:
                rows.append({"so_name": so, "jalmitra": jm})
        base_jm = pd.DataFrame(rows)
        grouped = base_jm.merge(grouped, on=["so_name","jalmitra"], how="left").fillna({"days_updated":0,"total_water":0.0})
        grouped["days_updated"] = grouped["days_updated"].astype(int)
        grouped["total_water"] = grouped["total_water"].astype(float).round(2)

        # normalize per-SO by max total of a jalmitra in that SO
        so_max = grouped.groupby("so_name")["total_water"].max().reset_index().rename(columns={"total_water":"so_max_total"})
        grouped = grouped.merge(so_max, on="so_name", how="left")
        grouped["so_max_total"] = grouped["so_max_total"].replace({0:np.nan})
        grouped["qty_norm"] = (grouped["total_water"] / grouped["so_max_total"]).fillna(0.0)
        grouped["days_norm"] = grouped["days_updated"] / float(period_days)
        grouped["jal_score"] = (0.5 * grouped["days_norm"] + 0.5 * grouped["qty_norm"]).round(4)
        grouped["jal_score"] = grouped["jal_score"].fillna(0.0)

        # per-SO aggregates
        so_metrics = grouped.groupby("so_name").agg(
            so_score = ("jal_score", "mean"),
            mean_days_updated = ("days_updated", "mean"),
            total_water_so = ("total_water","sum"),
            n_jalmitras = ("jalmitra", "nunique")
        ).reset_index()
        so_metrics["so_score"] = so_metrics["so_score"].fillna(0.0).round(4)
        so_metrics["mean_days_updated"] = so_metrics["mean_days_updated"].round(2)
        so_metrics["total_water_so"] = so_metrics["total_water_so"].round(2)
        return grouped, so_metrics

    jal_df_all, so_metrics = compute_jalmitra_metrics_for_period(st.session_state["readings"], st.session_state["schemes"], period)

    # If no metrics, let user know
    if so_metrics.empty:
        st.info("No readings available for the selected period. Generate multi-SO demo (in-page or sidebar).")
    else:
        # Add requested columns: Total No of Scheme, Functional, Non-Functional, Present Jalmitra (today), Schemes Updated (last N days)
        schemes_df = st.session_state["schemes"]
        readings_df = st.session_state["readings"]
        today_iso = datetime.date.today().isoformat()

        # precompute maps for efficiency
        total_schemes_map = schemes_df.groupby("so_name")["id"].nunique().to_dict() if not schemes_df.empty else {}
        func_schemes_map = schemes_df[schemes_df["functionality"]=="Functional"].groupby("so_name")["id"].nunique().to_dict() if not schemes_df.empty else {}
        # non-functional = total - functional
        present_jm_today_map = {}
        schemes_updated_map = {}

        # compute present jalmitras today per SO
        if not readings_df.empty:
            today_reads = readings_df[readings_df["reading_date"] == today_iso]
            for so in so_metrics["so_name"].tolist():
                present_jm_today_map[so] = int(today_reads[today_reads["so_name"] == so]["jalmitra"].nunique())
        else:
            for so in so_metrics["so_name"].tolist():
                present_jm_today_map[so] = 0

        # compute schemes updated in window per SO
        start_window = (datetime.date.today() - datetime.timedelta(days=period-1)).isoformat()
        end_window = datetime.date.today().isoformat()
        if not readings_df.empty:
            window_reads = readings_df[(readings_df["reading_date"] >= start_window) & (readings_df["reading_date"] <= end_window)]
            for so in so_metrics["so_name"].tolist():
                schemes_updated_map[so] = int(window_reads[window_reads["so_name"] == so]["scheme_id"].nunique())
        else:
            for so in so_metrics["so_name"].tolist():
                schemes_updated_map[so] = 0

        # add columns to so_metrics
        so_metrics["Total Schemes"] = so_metrics["so_name"].apply(lambda x: int(total_schemes_map.get(x, 0)))
        so_metrics["Functional Schemes"] = so_metrics["so_name"].apply(lambda x: int(func_schemes_map.get(x, 0)))
        so_metrics["Non-Functional Schemes"] = so_metrics.apply(lambda row: int(row["Total Schemes"] - row["Functional Schemes"]), axis=1)
        so_metrics["Present Jalmitra (Today)"] = so_metrics["so_name"].apply(lambda x: int(present_jm_today_map.get(x, 0)))
        # Schemes Updated in last N days (<= total schemes)
        so_metrics[f"Schemes Updated (last {period}d)"] = so_metrics["so_name"].apply(lambda x: int(min(schemes_updated_map.get(x, 0), total_schemes_map.get(x, 0))))
        so_metrics["Score of SO"] = so_metrics["so_score"]

        # Prepare Top 7 / Worst 7 (sort by Score of SO)
        so_metrics = so_metrics.sort_values(by="Score of SO", ascending=False).reset_index(drop=True)
        so_metrics.insert(0, "Rank", range(1, len(so_metrics)+1))
        top7 = so_metrics.head(7).copy()
        worst7 = so_metrics.tail(7).sort_values(by="Score of SO", ascending=True).reset_index(drop=True)

        # Select columns order for display
        display_cols = ["Rank","so_name","Total Schemes","Functional Schemes","Non-Functional Schemes",
                        "Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"]
        # rename so_name -> SO Name for display
        top7_display = top7[display_cols].rename(columns={"so_name":"SO Name"})
        worst7_display = worst7[display_cols].rename(columns={"so_name":"SO Name"})

        st.markdown("#### 🟢 Top 7 Performing SOs")
        st.dataframe(top7_display.style.format({"Score of SO":"{:.3f}"}).background_gradient(subset=["Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"], cmap="Greens"), use_container_width=True, height=320)

        st.markdown("#### 🔴 Worst 7 Performing SOs")
        st.dataframe(worst7_display.style.format({"Score of SO":"{:.3f}"}).background_gradient(subset=["Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"], cmap="Reds_r"), use_container_width=True, height=320)

        st.markdown("---")
        st.subheader("Open an SO Dashboard (click a name below)")
        leftcol, rightcol = st.columns(2)
        with leftcol:
            st.markdown("Top 7 — click to open (single click)")
            for _, r in top7.iterrows():
                nm = r["so_name"]
                if st.button(f"{int(r['Rank'])}. {nm}", key=f"aee_open_top_{nm}_{period}"):
                    st.session_state["selected_so_from_aee"] = nm
                    st.experimental_rerun()
        with rightcol:
            st.markdown("Worst 7 — click to open (single click)")
            for _, r in worst7.iterrows():
                nm = r["so_name"]
                if st.button(f"{int(r['Rank'])}. {nm}", key=f"aee_open_worst_{nm}_{period}"):
                    st.session_state["selected_so_from_aee"] = nm
                    st.experimental_rerun()

    st.stop()

# --------------------------- Section Officer dashboard (original page) ---------------------------

# If AEE set a chosen SO, use that; else default to ROKI RAY
chosen_so = None
if st.session_state.get("selected_so_from_aee"):
    chosen_so = st.session_state.pop("selected_so_from_aee")

if chosen_so:
    so = chosen_so
else:
    so = "ROKI RAY"

today = datetime.date.today()
st.header(f"Section Officer Dashboard — {so}")
st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")

schemes = st.session_state["schemes"]
readings = st.session_state["readings"]
if schemes.empty:
    st.info("No schemes found. Generate demo data first (single-SO demo from Section Officer view or multi-SO from AEE).")
    st.stop()

# SAFE merged_all to avoid KeyError
r = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"])
s = ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])

merged_all = r.merge(
    s[["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"]],
    left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme")
)

if "so_name_reading" in merged_all.columns or "so_name_scheme" in merged_all.columns:
    merged_all["so_name"] = merged_all.get("so_name_reading").combine_first(merged_all.get("so_name_scheme"))
else:
    merged_all["so_name"] = merged_all.get("so_name", "")

if "scheme_name_reading" in merged_all.columns or "scheme_name_scheme" in merged_all.columns:
    merged_all["Scheme Display"] = merged_all.get("scheme_name_reading").combine_first(merged_all.get("scheme_name_scheme"))
else:
    merged_all["Scheme Display"] = merged_all.get("scheme_name", "")

today_iso = datetime.date.today().isoformat()
today_upd = merged_all[
    (merged_all.get("reading_date", "") == today_iso) &
    (merged_all.get("functionality", "") == "Functional") &
    (merged_all.get("so_name", "") == so)
].copy()

if today_upd.shape[0] > 0 and "Scheme Display" not in today_upd.columns:
    if "scheme_label" in today_upd.columns:
        today_upd["Scheme Display"] = today_upd["scheme_label"]
    else:
        today_upd["Scheme Display"] = today_upd.get("scheme_name", "")
today_upd["Scheme Display"] = today_upd["Scheme Display"].fillna("").astype(str)

# Overview pies
func_counts = schemes["functionality"].value_counts()
updated_count = int(today_upd["jalmitra"].nunique()) if not today_upd.empty else 0
total_functional = int(len(schemes[schemes["functionality"] == "Functional"]))
absent_count = max(total_functional - updated_count, 0)

if st.session_state.get("view_mode","Web View") == "Web View":
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

# Rankings (SO page)
st.subheader("🏅 Jalmitra Performance — Top & Worst")
period = st.selectbox("Show performance for", [7, 15, 30], index=0, format_func=lambda x: f"{x} days", key="so_period")
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

    villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    rnd = random.Random(42)
    metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]
    metrics["ideal_total_Nd"] = metrics.get("ideal_total_Nd", 0.0).round(2)

    top_table = metrics.head(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
    top_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

    worst_table = metrics.sort_values(by='score', ascending=True).head(10)[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
    worst_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

    col_t, col_w = st.columns([1,1])
    with col_t:
        st.markdown(f"### 🟢 Top 10 Performing Jalmitras — last {period} days")
        st.dataframe(top_table.style.format({"Total Water (m³)":"{:.2f}","Ideal Water (m³)":"{:.2f}","Score":"{:.3f}"}).background_gradient(subset=[f"Days Updated (last {period}d)","Total Water (m³)","Score"], cmap="Greens"), height=360)
        st.download_button("⬇️ Download Top 10 CSV", top_table.to_csv(index=False).encode("utf-8"), "top_10_jalmitras.csv")
    with col_w:
        st.markdown(f"### 🔴 Worst 10 Performing Jalmitras — last {period} days")
        st.dataframe(worst_table.style.format({"Total Water (m³)":"{:.2f}","Ideal Water (m³)":"{:.2f}","Score":"{:.3f}"}).background_gradient(subset=[f"Days Updated (last {period}d)","Total Water (m³)","Score"], cmap="Reds_r"), height=360)
        st.download_button("⬇️ Download Worst 10 CSV", worst_table.to_csv(index=False).encode("utf-8"), "worst_10_jalmitras.csv")

    st.markdown("**Tap a name below to open the performance chart**")
    top_names = top_table["Jalmitra"].tolist()
    worst_names = worst_table["Jalmitra"].tolist()

    if st.session_state.get("view_mode","Web View") == "Web View":
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

# Show performance chart when name selected
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")

    # recompute last window for selected period
    last_window, _ = compute_metrics(readings, schemes, so, start_date, end_date)
    jm_data = last_window[last_window["jalmitra"] == jm] if (not last_window.empty) else pd.DataFrame()

    st.subheader(f"Performance — {jm}")
    if jm_data.empty:
        st.info("No readings for this Jalmitra in the selected window.")
    else:
        # prepare full date list (inclusive) for the selected period
        dates = [(datetime.date.today() - datetime.timedelta(days=d)).isoformat() for d in reversed(range(period))]
        # aggregate actual per date
        agg_by_date = jm_data.groupby("reading_date")["water_quantity"].sum().reset_index().rename(columns={"reading_date":"Date","water_quantity":"Water (m³)"})
        # ensure every date exists
        full_dates_df = pd.DataFrame({"Date": dates})
        daily = full_dates_df.merge(agg_by_date, on="Date", how="left").fillna({"Water (m³)":0.0})
        daily["Water (m³)"] = daily["Water (m³)"].astype(float).round(2)

        # compute ideal per day for this jalmitra (sum of ideal_per_day across distinct schemes he covers)
        scheme_col = "Scheme Display" if "Scheme Display" in jm_data.columns else ("scheme_name" if "scheme_name" in jm_data.columns else "scheme_label")
        jm_schemes = jm_data[[ "scheme_id", scheme_col, "ideal_per_day"]].drop_duplicates(subset=["scheme_id"])
        scheme_names = jm_schemes[scheme_col].astype(str).unique().tolist() if not jm_schemes.empty else []
        scheme_names_display = ", ".join(scheme_names) if scheme_names else "—"
        ideal_per_day_sum = jm_schemes["ideal_per_day"].sum() if not jm_schemes.empty else 0.0
        ideal_series = [round(float(ideal_per_day_sum), 2) for _ in dates]

        # determine bar colors: green when actual >= ideal, red otherwise
        bar_colors = []
        for actual, ideal in zip(daily["Water (m³)"].tolist(), ideal_series):
            if actual >= ideal:
                bar_colors.append("green")
            else:
                bar_colors.append("red")

        # build plotly figure: colored bars for actual, red line for ideal (stretched full width)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["Date"], y=daily["Water (m³)"], name="Actual Water (m³)", marker_color=bar_colors))
        # ideal as a continuous line across full date range
        fig.add_trace(go.Scatter(x=dates, y=ideal_series,
                                 mode="lines+markers",
                                 name="Ideal (m³/day)",
                                 line=dict(color="red", width=3),
                                 marker=dict(size=6)))
        # ensure x-axis covers full categorical domain and ideal line spans left-to-right
        fig.update_layout(
            title=f"{jm} — Daily Water Supplied (Last {period} Days) — Schemes: {scheme_names_display}",
            xaxis_title="Date",
            yaxis_title="Water (m³)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=40, r=20, t=80, b=40),
            height=420,
            xaxis=dict(type="category", categoryorder="array", categoryarray=dates)
        )
        st.plotly_chart(fig, use_container_width=True, height=420)

        # display totals and comparison
        total_actual = daily["Water (m³)"].sum()
        total_ideal = sum(ideal_series)
        days_updated = (daily["Water (m³)"] > 0).sum()
        st.markdown(f"**Total ({period} days):** {total_actual:.2f} m³  **Days Updated:** {days_updated}/{period}")
        st.markdown(f"**Ideal total ({period} days)**: {total_ideal:.2f} m³ — *(ideal per day: {ideal_per_day_sum:.2f} m³)*")

        # Download raw readings for this jalmitra for selected period — include zero days
        # Build export dataframe: Date, Time, Scheme Name, BFM Reading, Water (m³)
        # For days with multiple readings we will include each row; but to ensure zeros days present we will also create rows with zeros.
        # Approach:
        #  - take jm_data rows within window, keep Date, Time, Scheme Display, reading, water_quantity
        #  - then create a "dates skeleton" for the period and ensure dates with no records produce a single zero row
        jm_rows = jm_data[[ "reading_date", "reading_time", scheme_col, "reading", "water_quantity" ]].copy()
        jm_rows = jm_rows.rename(columns={ "reading_date":"Date", "reading_time":"Time", scheme_col:"Scheme Name", "reading":"BFM Reading", "water_quantity":"Water (m³)" })
        # ensure Date is in the same order and string format
        jm_rows["Date"] = jm_rows["Date"].astype(str)

        # aggregate by Date+Time+Scheme if multiple readings exist on same day; keep them all (we'll include zeros rows separately)
        export_rows = jm_rows.copy()

        # Now ensure zero-day rows exist: for any date in 'dates' that is not present in export_rows.Date, add a single row with zeros/blanks
        present_dates = set(export_rows["Date"].unique().tolist())
        zero_rows = []
        for d in dates:
            if d not in present_dates:
                zero_rows.append({"Date": d, "Time":"", "Scheme Name":"", "BFM Reading":"", "Water (m³)":0.0})
        if zero_rows:
            export_rows = pd.concat([export_rows, pd.DataFrame(zero_rows)], ignore_index=True)

        export_rows = export_rows.sort_values("Date").reset_index(drop=True)
        # Format numeric columns
        if "Water (m³)" in export_rows.columns:
            export_rows["Water (m³)"] = pd.to_numeric(export_rows["Water (m³)"], errors="coerce").fillna(0.0).round(2)
        csv_bytes = export_rows.to_csv(index=False).encode("utf-8")
        st.download_button(f"⬇️ Download {jm} readings — last {period} days (CSV)", csv_bytes, file_name=f"{jm}_readings_{period}d.csv", mime="text/csv")

    if st.session_state.get("view_mode","Web View") == "Web View":
        if st.button("Close View"):
            st.session_state["selected_jalmitra"] = None
    else:
        if st.button("Close View (Phone)"):
            st.session_state["selected_jalmitra"] = None

st.markdown("---")

# BFM Readings Updated Today
st.subheader("📅 BFM Readings Updated Today")
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    today_upd["Scheme Name"] = today_upd.get("Scheme Display", "")
    today_upd["reading"] = pd.to_numeric(today_upd.get("reading", 0), errors="coerce").fillna(0).astype(int)
    today_upd["water_quantity"] = pd.to_numeric(today_upd.get("water_quantity", 0.0), errors="coerce").fillna(0.0).round(2)
    today_upd["BFM Reading Display"] = today_upd["reading"].apply(lambda x: f"{x:06d}")
    daily_bfm = today_upd[["jalmitra","Scheme Name","BFM Reading Display","reading_time","water_quantity"]].copy() if "Scheme Name" in today_upd.columns else today_upd[["jalmitra","Scheme Display","BFM Reading Display","reading_time","water_quantity"]].copy()
    # ensure consistent column name
    if "Scheme Name" not in daily_bfm.columns and "Scheme Display" in daily_bfm.columns:
        daily_bfm = daily_bfm.rename(columns={"Scheme Display":"Scheme Name"})
    daily_bfm.columns = ["Jalmitra","Scheme Name","BFM Reading","Reading Time","Water Quantity (m³)"]
    daily_bfm = daily_bfm.sort_values("Jalmitra").reset_index(drop=True)
    daily_bfm.insert(0, "S.No", range(1, len(daily_bfm)+1))
    try:
        sty = daily_bfm.style.format({"Water Quantity (m³)":"{:.2f}"})
        st.dataframe(sty.background_gradient(cmap="Blues", subset=["Water Quantity (m³)"]), height=360)
    except Exception:
        st.dataframe(daily_bfm, height=360)

# Export
st.markdown("---")
st.subheader("📤 Export Snapshot")
st.download_button("Schemes CSV", schemes.to_csv(index=False).encode("utf-8"), "schemes.csv")
st.download_button("Readings CSV", readings.to_csv(index=False).encode("utf-8"), "readings.csv")
try:
    st.download_button("Metrics CSV", metrics.to_csv(index=False).encode("utf-8"), "metrics.csv")
except Exception:
    st.info("Metrics CSV not available.")

st.success(f"Dashboard ready for SO {so}. Demo data generated: {st.session_state['demo_generated']}")
