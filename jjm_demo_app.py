# jjm_demo_app.py
# JJM Dashboard — Unified (Fixed) + Executive Engineer dashboard + restored SO sections
# Replace your existing file with this full file.

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import random
import plotly.express as px

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Unified (Fixed)", layout="wide")
try:
    st.image("logo.jpg", width=160)
except Exception:
    pass
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
    st.session_state.setdefault("jalmitras_map", {})         # so_name -> list of jalmitras
    st.session_state.setdefault("scheme_jalmitra_map", {})  # so_name -> {scheme_id: jalmitra}
    st.session_state.setdefault("jalmitra_scheme_map", {})  # so_name -> {jalmitra: scheme_label}
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)
    st.session_state.setdefault("selected_so_from_aee", None)
    st.session_state.setdefault("view_mode", "Web View")
    # Executive demo containers (separate)
    st.session_state.setdefault("exec_schemes", pd.DataFrame())
    st.session_state.setdefault("exec_readings", pd.DataFrame())
    st.session_state.setdefault("exec_demo_generated", False)

init_state()

# --------------------------- Demo generation & reset -----------
def reset_session_data():
    st.session_state["schemes"] = pd.DataFrame(columns=["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])
    st.session_state["readings"] = pd.DataFrame(columns=[
        "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"])
    st.session_state["jalmitras_map"] = {}
    st.session_state["scheme_jalmitra_map"] = {}
    st.session_state["jalmitra_scheme_map"] = {}
    st.session_state["next_scheme_id"] = 1
    st.session_state["next_reading_id"] = 1
    st.session_state["demo_generated"] = False
    st.session_state["selected_jalmitra"] = None
    st.session_state["selected_so_from_aee"] = None

def generate_demo_data(total_schemes:int=20, so_name:str="ROKI RAY"):
    """
    Generate demo for single SO:
      - creates `total_schemes` schemes
      - assigns one jalmitra per scheme (unique)
      - stores mappings under st.session_state keyed by so_name
      - generates readings for last 30 days (only for Functional schemes)
      - per-jalmitra update probability between 10% - 95%
    """
    assamese = [
        "Biren","Nagen","Rahul","Vikram","Debojit","Anup","Kamal","Ranjit","Himangshu",
        "Pranjal","Rupam","Dilip","Utpal","Amit","Jayanta","Hemanta","Rituraj","Dipankar",
        "Bikash","Dhruba","Subham","Pritam","Saurav","Bijoy","Manoj","Rupen","Kumar"
    ]
    villages = [
        "Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar",
        "Jorhat","Hajo","Tihu","Kokrajhar","Nalbari","Barpeta","Rangia","Goalpara","Dhemaji",
        "Dibrugarh","Mariani","Sonari"
    ]
    today = datetime.date.today()
    schemes = []
    readings = []
    scheme_jalmitra_map = {}
    jalmitra_scheme_map = {}
    jalmitras = []
    jalmitra_probs = {}

    # create unique jalmitra names (one per scheme)
    for i in range(total_schemes):
        base = assamese[i % len(assamese)]
        jm_name = f"{base}_{i+1}"
        jalmitras.append(jm_name)
        jalmitra_probs[jm_name] = round(random.uniform(0.10, 0.95), 3)

    reading_samples = [110010,215870,150340,189420,200015,234870]

    # create schemes and map unique jalmitra to scheme
    sid_start = st.session_state.get("next_scheme_id", 1)
    for i in range(total_schemes):
        sid = sid_start + i
        ideal_per_day = round(random.uniform(20.0, 100.0), 2)
        scheme_label = random.choice(villages) + " PWSS"
        functionality = random.choice(["Functional","Non-Functional"])
        schemes.append({
            "id": sid,
            "scheme_name": f"Scheme {chr(65 + (i % 26))}{'' if i < 26 else i//26}",
            "functionality": functionality,
            "so_name": so_name,
            "ideal_per_day": ideal_per_day,
            "scheme_label": scheme_label
        })
        assigned_jm = jalmitras[i]
        scheme_jalmitra_map[sid] = assigned_jm
        jalmitra_scheme_map[assigned_jm] = scheme_label

    # generate readings for functional schemes for last 30 days
    rid = st.session_state.get("next_reading_id", 1)
    days_to_generate = 30
    for s in schemes:
        if s["functionality"] != "Functional":
            continue
        assigned_jm = scheme_jalmitra_map[s["id"]]
        jm_prob = jalmitra_probs.get(assigned_jm, 0.5)
        for d in range(days_to_generate):
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < jm_prob:
                hour = random.randint(6, 11)
                minute = random.choice([0,15,30,45])
                time_str = f"{hour}:{minute:02d} AM"
                water_qty = round(random.uniform(10.0, 100.0), 2)
                readings.append({
                    "id": rid,
                    "scheme_id": s["id"],
                    "jalmitra": assigned_jm,
                    "reading": random.choice(reading_samples),
                    "reading_date": date_iso,
                    "reading_time": time_str,
                    "water_quantity": water_qty,
                    "scheme_name": s["scheme_label"],
                    "so_name": so_name
                })
                rid += 1

    # merge new data into session
    schemes_df = st.session_state["schemes"]
    readings_df = st.session_state["readings"]

    new_schemes_df = pd.DataFrame(schemes)
    new_readings_df = pd.DataFrame(readings)

    if not new_schemes_df.empty:
        schemes_df = pd.concat([schemes_df, new_schemes_df], ignore_index=True)
    if not new_readings_df.empty:
        readings_df = pd.concat([readings_df, new_readings_df], ignore_index=True)

    st.session_state["schemes"] = schemes_df.reset_index(drop=True)
    st.session_state["readings"] = readings_df.reset_index(drop=True)
    st.session_state["jalmitras_map"][so_name] = jalmitras
    st.session_state["scheme_jalmitra_map"].setdefault(so_name, {})
    st.session_state["scheme_jalmitra_map"][so_name].update(scheme_jalmitra_map)
    st.session_state["jalmitra_scheme_map"].setdefault(so_name, {})
    st.session_state["jalmitra_scheme_map"][so_name].update(jalmitra_scheme_map)
    st.session_state["next_scheme_id"] = sid_start + total_schemes
    st.session_state["next_reading_id"] = rid
    st.session_state["demo_generated"] = True
    st.success(f"✅ Demo data generated for {so_name}.")

def generate_multi_so_demo(num_sos=14, schemes_per_so=18, max_days=30):
    """
    Generate multi-SO demo for AEE view and store per-SO mappings.
    """
    random.seed(42)
    base_so_names = [
        "ROKI RAY", "Sanjay Das", "Anup Bora", "Ranjit Kalita", "Bikash Deka", "Manoj Das",
        "Dipankar Nath", "Himangshu Deka", "Kamal Choudhury", "Rituraj Das", "Debojit Gogoi",
        "Utpal Saikia", "Pritam Bora", "Amit Baruah", "Sunil Kumar", "Raju Das"
    ][:max(1, num_sos)]

    schemes_rows = []
    readings_rows = []
    jalmitras_map = {}
    scheme_jalmitra_map_all = {}
    jalmitra_scheme_map_all = {}
    sid = st.session_state.get("next_scheme_id", 1)
    rid = st.session_state.get("next_reading_id", 1)
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

    for so_index, so in enumerate(base_so_names[:num_sos]):
        jm_list = []
        for i in range(schemes_per_so):
            jm_name = f"{assamese[(so_index*schemes_per_so + i) % len(assamese)]}_{so_index+1}_{i+1}"
            jm_list.append(jm_name)
        jalmitras_map[so] = jm_list
        scheme_jalmitra_map_all[so] = {}
        jalmitra_scheme_map_all[so] = {}

        for i in range(schemes_per_so):
            ideal_per_day = round(random.uniform(20.0, 100.0), 2)
            scheme_label = random.choice(villages) + " PWSS"
            func = "Functional" if random.random() > 0.25 else "Non-Functional"
            schemes_rows.append({
                "id": sid,
                "scheme_name": f"Scheme_{sid}_{so.split()[0]}",
                "functionality": func,
                "so_name": so,
                "ideal_per_day": ideal_per_day,
                "scheme_label": scheme_label
            })
            assigned_jm = jm_list[i]
            scheme_jalmitra_map_all[so][sid] = assigned_jm
            jalmitra_scheme_map_all[so][assigned_jm] = scheme_label
            sid += 1

    # Generate readings for each SO's functional schemes
    for so in jalmitras_map.keys():
        so_schemes = [r for r in schemes_rows if r["so_name"] == so and r["functionality"] == "Functional"]
        so_scheme_ids = [s["id"] for s in so_schemes]
        jm_list = jalmitras_map[so]
        for jm in jm_list:
            jm_rng = random.Random(abs(hash(so + jm)) % (2**32))
            jm_prob = jm_rng.uniform(0.10, 0.95)
            for d in range(max_days):
                date_iso = (today - datetime.timedelta(days=d)).isoformat()
                if jm_rng.random() < jm_prob and so_scheme_ids:
                    hour = jm_rng.randint(6, 11)
                    minute = jm_rng.choice([0,15,30,45])
                    time_str = f"{hour}:{minute:02d} AM"
                    water_qty = round(jm_rng.uniform(10.0, 100.0), 2)
                    readings_rows.append({
                        "id": rid,
                        "scheme_id": jm_rng.choice(so_scheme_ids),
                        "jalmitra": jm,
                        "reading": jm_rng.choice([110010,215870,150340,189420,200015,234870]),
                        "reading_date": date_iso,
                        "reading_time": time_str,
                        "water_quantity": water_qty,
                        "so_name": so
                    })
                    rid += 1

    schemes_df_new = pd.DataFrame(schemes_rows)
    readings_df_new = pd.DataFrame(readings_rows)

    # attach scheme_label into readings_df_new where possible
    if not readings_df_new.empty and not schemes_df_new.empty:
        readings_df_new = readings_df_new.merge(schemes_df_new[["id","scheme_label"]], left_on="scheme_id", right_on="id", how="left")
        if "scheme_label" in readings_df_new.columns:
            readings_df_new["scheme_name"] = readings_df_new["scheme_label"]
            readings_df_new.drop(columns=["scheme_label"], inplace=True, errors="ignore")

    # Append to session
    if not schemes_df_new.empty:
        st.session_state["schemes"] = pd.concat([st.session_state["schemes"], schemes_df_new], ignore_index=True)
    if not readings_df_new.empty:
        st.session_state["readings"] = pd.concat([st.session_state["readings"], readings_df_new], ignore_index=True)

    # merge mapping dicts into session per SO
    for so in jalmitras_map:
        st.session_state["jalmitras_map"].setdefault(so, [])
        st.session_state["jalmitras_map"][so] = jalmitras_map[so]
        st.session_state["scheme_jalmitra_map"].setdefault(so, {})
        st.session_state["scheme_jalmitra_map"][so].update(scheme_jalmitra_map_all.get(so, {}))
        st.session_state["jalmitra_scheme_map"].setdefault(so, {})
        st.session_state["jalmitra_scheme_map"][so].update(jalmitra_scheme_map_all.get(so, {}))

    st.session_state["next_scheme_id"] = sid
    st.session_state["next_reading_id"] = rid
    st.session_state["demo_generated"] = True
    st.success("✅ Multi-SO demo generated for AEE.")

# --------------------------- compute_metrics (robust) -----------
@st.cache_data
def compute_metrics(readings: pd.DataFrame, schemes: pd.DataFrame, so: str, start: str, end: str):
    r = ensure_columns(readings.copy() if readings is not None else pd.DataFrame(), [
        "id", "scheme_id", "jalmitra", "reading", "reading_date", "reading_time", "water_quantity", "scheme_name", "so_name"
    ])
    s = ensure_columns(schemes.copy() if schemes is not None else pd.DataFrame(), [
        "id", "scheme_name", "functionality", "so_name", "ideal_per_day", "scheme_label"
    ])

    merged = r.merge(
        s[["id", "scheme_name", "functionality", "so_name", "ideal_per_day", "scheme_label"]],
        left_on="scheme_id", right_on="id", how="left", suffixes=("_reading", "_scheme")
    )

    def col_or_empty(df, colname, fallback=""):
        if colname in df.columns:
            return df[colname]
        else:
            return pd.Series([fallback] * len(df), index=df.index)

    so_reading = col_or_empty(merged, "so_name_reading")
    so_scheme = col_or_empty(merged, "so_name_scheme")
    merged["so_name"] = so_reading.fillna("").replace("", np.nan).fillna(so_scheme.fillna("").astype(str)).fillna("").astype(str)

    scheme_name_reading = col_or_empty(merged, "scheme_name_reading")
    scheme_name_scheme = col_or_empty(merged, "scheme_name_scheme")
    merged["Scheme Display"] = scheme_name_reading.combine_first(scheme_name_scheme).fillna(merged.get("scheme_name", ""))

    merged["functionality"] = col_or_empty(merged, "functionality").fillna("")
    merged["ideal_per_day"] = pd.to_numeric(col_or_empty(merged, "ideal_per_day", 0.0), errors="coerce").fillna(0.0)

    mask = (
        (merged.get("functionality", "") == "Functional")
        & (merged.get("so_name", "") == so)
        & (merged.get("reading_date", "") >= start)
        & (merged.get("reading_date", "") <= end)
    )
    lastN = merged.loc[mask].copy()
    if lastN.empty:
        empty_metrics = pd.DataFrame(columns=["jalmitra", "days_updated", "total_water_m3", "schemes_covered", "ideal_total_Nd", "quantity_score"])
        return lastN, empty_metrics

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

    scheme_ideal = lastN[["jalmitra", "scheme_id", "ideal_per_day"]].drop_duplicates(subset=["jalmitra", "scheme_id"])
    scheme_ideal["ideal_Nd"] = pd.to_numeric(scheme_ideal["ideal_per_day"], errors="coerce").fillna(0.0) * float(days_count)
    ideal_sum = scheme_ideal.groupby("jalmitra")["ideal_Nd"].sum().reset_index().rename(columns={"ideal_Nd": "ideal_total_Nd"})

    metrics = agg.merge(ideal_sum, on="jalmitra", how="left")
    metrics["ideal_total_Nd"] = metrics["ideal_total_Nd"].fillna(0.0).round(2)

    def compute_qs(row):
        ideal = float(row.get("ideal_total_Nd", 0.0) or 0.0)
        water = float(row.get("total_water_m3", 0.0) or 0.0)
        if ideal <= 0:
            return 0.0
        return min(water / ideal, 1.0)

    metrics["quantity_score"] = metrics.apply(compute_qs, axis=1)
    metrics["days_updated"] = metrics["days_updated"].astype(int)
    metrics["total_water_m3"] = metrics["total_water_m3"].astype(float).round(2)
    metrics["quantity_score"] = metrics["quantity_score"].astype(float).round(3)
    metrics.attrs["days_count"] = days_count

    return lastN, metrics

# --------------------------- Sidebar & AEE demo controls ---------------------------
st.sidebar.header("Demo Controls")
if st.sidebar.button("Generate multi-SO demo (14 SOs)"):
    generate_multi_so_demo(num_sos=14, schemes_per_so=18, max_days=30)
if st.sidebar.button("Clear demo data (sidebar)"):
    reset_session_data()
    st.sidebar.warning("Session demo data cleared (sidebar).")
st.sidebar.markdown("---")
st.sidebar.write("SO demo generator available on the Section Officer page.")

# --------------------------- Top: role & view ---------------------------
_roles = ["Section Officer", "Assistant Executive Engineer", "Executive Engineer"]
col_r1, col_r2 = st.columns([2,1])
with col_r1:
    role = st.selectbox("Select Role", _roles, index=0, key="role_widget")
with col_r2:
    st.session_state["view_mode"] = st.radio("View Mode", ["Web View", "Phone View"], horizontal=True, key="view_widget")
st.markdown("---")

# --------------------------- AEE page -----------
if role == "Assistant Executive Engineer":
    st.header("Assistant Executive Engineer Dashboard (Aggregated from SOs)")
    st.markdown(f"**AEE:** Er. ROKI RAY  •  **Subdivision:** Guwahati")
    st.markdown(f"**DATE:** {datetime.date.today().strftime('%A, %d %B %Y').upper()}")
    st.markdown("---")

    st.markdown("#### AEE demo controls (generate or remove SOs under this AEE)")
    ac1, ac2, ac3 = st.columns([2,1,1])
    with ac1:
        aee_num_sos = st.number_input("Number of SOs to generate", min_value=1, max_value=30, value=14, key="aee_num_sos")
        aee_schemes_per_so = st.number_input("Schemes per SO", min_value=4, max_value=50, value=18, key="aee_schemes_per_so")
    with ac2:
        if st.button("Generate AEE demo (in-page)", key="btn_aee_gen_inpage"):
            generate_multi_so_demo(num_sos=int(aee_num_sos), schemes_per_so=int(aee_schemes_per_so), max_days=30)
    with ac3:
        if st.button("Remove AEE demo (in-page)", key="btn_aee_rem_inpage"):
            reset_session_data()
            st.success("✅ AEE demo removed from session (in-page).")

    st.markdown("---")
    if not st.session_state["demo_generated"]:
        st.info("For AEE view, generate multi-SO demo data using the buttons above (or use the sidebar).")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Scheme Functionality (all SOs)")
        if not st.session_state["schemes"].empty:
            schemes_df_all = st.session_state.get("schemes", pd.DataFrame())
            total_schemes = len(schemes_df_all)
            num_func = int(schemes_df_all[schemes_df_all["functionality"] == "Functional"]["id"].nunique())
            num_non = max(total_schemes - num_func, 0)
            func_counts = pd.Series({"Functional": num_func, "Non-Functional": num_non})
            st.markdown(f"<small>Functional: <b>{num_func}</b> • Non-Functional: <b>{num_non}</b> • Total: <b>{total_schemes}</b></small>", unsafe_allow_html=True)
            fig = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                         color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
            fig.update_traces(textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True, height=260)
        else:
            st.write("No schemes available.")
    with col2:
        st.subheader("SO Updates (today)")
        today_iso = datetime.date.today().isoformat()
        if not st.session_state["readings"].empty and st.session_state["demo_generated"]:
            today_updates = st.session_state["readings"][st.session_state["readings"]["reading_date"] == today_iso]
            total_updates = int(today_updates["jalmitra"].nunique())
            total_functional = int(len(st.session_state["schemes"][st.session_state["schemes"]["functionality"] == "Functional"]))
            df_upd = pd.DataFrame({"status":["Updated today (unique jalmitras)","Other (approx)"], "count":[total_updates, max(total_functional - total_updates, 0)]})
            st.markdown(f"<small>Present: <b>{total_updates}</b> • Other (approx): <b>{max(total_functional - total_updates, 0)}</b></small>", unsafe_allow_html=True)
            fig2 = px.pie(df_upd, names="status", values="count", color="status",
                          color_discrete_map={"Updated today (unique jalmitras)":"#4CAF50","Other (approx)":"#F44336"})
            fig2.update_traces(textinfo='percent+label')
            st.plotly_chart(fig2, use_container_width=True, height=260)
        else:
            st.info("No readings available for today. Generate AEE demo to populate data.")

    st.markdown("---")
    st.subheader("Section Officer performance (aggregated from Jalmitra scores)")
    period = st.selectbox("Select window (days)", [7,15,30], index=0, key="aee_period")
    st.markdown(f"Showing performance for last **{period} days**")

    def compute_jalmitra_metrics_for_period(readings_df, schemes_df, period_days):
        start_date = (datetime.date.today() - datetime.timedelta(days=period_days-1)).isoformat()
        end_date = datetime.date.today().isoformat()
        sel = readings_df[(readings_df["reading_date"] >= start_date) & (readings_df["reading_date"] <= end_date)].copy() if not readings_df.empty else pd.DataFrame()
        rows = []
        for so_key, jlist in st.session_state.get("jalmitras_map", {}).items():
            for jm in jlist:
                rows.append({"so_name": so_key, "jalmitra": jm})
        base_jm = pd.DataFrame(rows)
        if base_jm.empty and not sel.empty:
            base_jm = sel[["so_name","jalmitra"]].drop_duplicates().rename(columns={"so_name":"so_name","jalmitra":"jalmitra"})
        if base_jm.empty:
            return pd.DataFrame(), pd.DataFrame()

        grouped = sel.groupby(["so_name","jalmitra"]).agg(
            days_updated = ("reading_date", lambda x: x.nunique()),
            total_water = ("water_quantity", "sum")
        ).reset_index() if not sel.empty else pd.DataFrame(columns=["so_name","jalmitra","days_updated","total_water"])

        grouped = base_jm.merge(grouped, on=["so_name","jalmitra"], how="left").fillna({"days_updated":0,"total_water":0.0})
        grouped["days_updated"] = grouped["days_updated"].astype(int)
        grouped["total_water"] = grouped["total_water"].astype(float).round(2)
        so_max = grouped.groupby("so_name")["total_water"].max().reset_index().rename(columns={"total_water":"so_max_total"})
        grouped = grouped.merge(so_max, on="so_name", how="left")
        grouped["so_max_total"] = grouped["so_max_total"].replace({0:np.nan})
        grouped["qty_norm"] = (grouped["total_water"] / grouped["so_max_total"]).fillna(0.0)
        grouped["days_norm"] = grouped["days_updated"] / float(period_days)
        grouped["jal_score"] = (0.5 * grouped["days_norm"] + 0.5 * grouped["qty_norm"]).round(4)
        grouped["jal_score"] = grouped["jal_score"].fillna(0.0)

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

    jal_df_all, so_metrics = compute_jalmitra_metrics_for_period(st.session_state.get("readings", pd.DataFrame()), st.session_state.get("schemes", pd.DataFrame()), period)

    if so_metrics.empty:
        st.info("No readings available for the selected period. Generate multi-SO demo.")
    else:
        schemes_df = st.session_state.get("schemes", pd.DataFrame())
        readings_df = st.session_state.get("readings", pd.DataFrame())

        total_schemes_map = schemes_df.groupby("so_name")["id"].nunique().to_dict() if not schemes_df.empty else {}
        func_schemes_map = schemes_df[schemes_df["functionality"]=="Functional"].groupby("so_name")["id"].nunique().to_dict() if not schemes_df.empty else {}
        present_jm_today_map = {}
        if not readings_df.empty and st.session_state["demo_generated"]:
            today_reads = readings_df[readings_df["reading_date"] == datetime.date.today().isoformat()]
            for so_key in so_metrics["so_name"].tolist():
                present_jm_today_map[so_key] = int(today_reads[today_reads["so_name"] == so_key]["jalmitra"].nunique())
        else:
            for so_key in so_metrics["so_name"].tolist():
                present_jm_today_map[so_key] = 0

        start_window = (datetime.date.today() - datetime.timedelta(days=period-1)).isoformat()
        end_window = datetime.date.today().isoformat()
        schemes_updated_map = {}
        if not readings_df.empty and st.session_state["demo_generated"]:
            window_reads = readings_df[(readings_df["reading_date"] >= start_window) & (readings_df["reading_date"] <= end_window)]
            for so_key in so_metrics["so_name"].tolist():
                schemes_updated_map[so_key] = int(window_reads[window_reads["so_name"] == so_key]["scheme_id"].nunique())
        else:
            for so_key in so_metrics["so_name"].tolist():
                schemes_updated_map[so_key] = 0

        so_metrics["Total Schemes"] = so_metrics["so_name"].apply(lambda x: int(total_schemes_map.get(x, 0)))
        so_metrics["Functional Schemes"] = so_metrics["so_name"].apply(lambda x: int(func_schemes_map.get(x, 0)))
        so_metrics["Non-Functional Schemes"] = so_metrics.apply(lambda row: int(row["Total Schemes"] - row["Functional Schemes"]), axis=1)
        so_metrics["Present Jalmitra (Today)"] = so_metrics["so_name"].apply(lambda x: int(present_jm_today_map.get(x, 0)))
        so_metrics[f"Schemes Updated (last {period}d)"] = so_metrics["so_name"].apply(lambda x: int(min(schemes_updated_map.get(x, 0), total_schemes_map.get(x, 0))))
        so_metrics["Score of SO"] = so_metrics["so_score"]
        so_metrics = so_metrics.sort_values(by="Score of SO", ascending=False).reset_index(drop=True)
        so_metrics.insert(0, "Rank", range(1, len(so_metrics)+1))
        top7 = so_metrics.head(7).copy()
        worst7 = so_metrics.tail(7).sort_values(by="Score of SO", ascending=True).reset_index(drop=True)

        display_cols = ["Rank","so_name","Total Schemes","Functional Schemes","Non-Functional Schemes",
                        "Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"]
        top7_display = top7[display_cols].rename(columns={"so_name":"SO Name"})
        worst7_display = worst7[display_cols].rename(columns={"so_name":"SO Name"})

        st.markdown("#### 🟢 Top 7 Performing SOs")
        st.dataframe(top7_display.style.format({"Score of SO":"{:.3f}"}).background_gradient(subset=["Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"], cmap="Greens"), use_container_width=True, height=320)
        st.markdown("#### 🔴 Worst 7 Performing SOs")
        st.dataframe(worst7_display.style.format({"Score of SO":"{:.3f}"}).background_gradient(subset=["Present Jalmitra (Today)", f"Schemes Updated (last {period}d)","Score of SO"], cmap="Reds_r"), use_container_width=True, height=320)

        st.markdown("---")

        # Open SO dashboard buttons (styled)
        st.subheader("Open an SO Dashboard (click a name below — opens a new tab)")
        st.markdown(
            """
            <style>
            .aee-open-btn { display:inline-block; margin:6px 0; }
            .aee-btn {
                font-weight:600;
                padding:10px 16px;
                border-radius:12px;
                border: none;
                cursor: pointer;
                text-decoration:none;
                color: #0f1724;
                box-shadow: 0 6px 18px rgba(5,10,25,0.35);
                transition: transform .12s ease, box-shadow .12s ease;
                display:inline-block;
            }
            .aee-btn:hover { transform: translateY(-4px); box-shadow: 0 12px 28px rgba(5,10,25,0.45); }
            .aee-btn-top { background: linear-gradient(135deg, #dff6ec 0%, #b9f0c7 50%, #7ee092 100%); color:#04221a; }
            .aee-btn-worst { background: linear-gradient(135deg, #ffecee 0%, #ffd7d7 50%, #ffc3c3 100%); color:#2b0b0b; }
            .aee-rank { background: rgba(255,255,255,0.18); padding:4px 8px; border-radius:8px; margin-right:8px; font-weight:700; font-size:0.9em; color: inherit; }
            </style>
            """, unsafe_allow_html=True
        )
        leftcol, rightcol = st.columns(2)
        with leftcol:
            st.markdown("Top 7 — click to open (new tab)")
            if not top7.empty:
                for idx, r in top7.iterrows():
                    nm = r["so_name"]
                    rank = int(r["Rank"])
                    url = f"?role=Section+Officer&so={nm.replace(' ','%20')}"
                    html = (
                        f'<div class="aee-open-btn">'
                        f'<a class="aee-btn aee-btn-top" href="{url}" target="_blank">'
                        f'<span class="aee-rank">{rank}</span> Open {rank}. {nm}'
                        f'</a></div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No Top entries.")
        with rightcol:
            st.markdown("Worst 7 — click to open (new tab)")
            if not worst7.empty:
                for idx, r in worst7.iterrows():
                    nm = r["so_name"]
                    rank = int(r["Rank"])
                    url = f"?role=Section+Officer&so={nm.replace(' ','%20')}"
                    html = (
                        f'<div class="aee-open-btn">'
                        f'<a class="aee-btn aee-btn-worst" href="{url}" target="_blank">'
                        f'<span class="aee-rank">{rank}</span> Open {rank}. {nm}'
                        f'</a></div>'
                    )
                    st.markdown(html, unsafe_allow_html=True)
            else:
                st.write("No Worst entries.")

# --------------------------- Executive Engineer dashboard (NEW) -----------
if role == "Executive Engineer":
    st.header("Executive Engineer Dashboard — Overview")
    st.markdown(f"**DATE:** {datetime.date.today().strftime('%A, %d %B %Y').upper()}")
    st.markdown("---")

    st.markdown("#### Executive demo controls (generate or clear Exec demo data)")
    ecol1, ecol2 = st.columns([2,1])
    with ecol1:
        exec_generate = st.button("Generate Exec Demo (300 schemes, 250 jalmitras)", key="btn_exec_gen")
    with ecol2:
        if st.button("Clear Exec Demo", key="btn_exec_clear"):
            st.session_state["exec_schemes"] = pd.DataFrame()
            st.session_state["exec_readings"] = pd.DataFrame()
            st.session_state["exec_demo_generated"] = False
            st.success("Exec demo cleared.")

    if exec_generate:
        total_schemes_exec = 300
        total_jalmitras_exec = 250
        eees = ["AEE ROKI", "AEE ANWAR"]
        schemes_rows = []
        readings_rows = []
        jm_list = [f"ExecJM_{i+1}" for i in range(total_jalmitras_exec)]
        villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo","Tihu","Nalbari"]
        sid = 100000
        rid = 500000
        today = datetime.date.today()

        for i in range(total_schemes_exec):
            aee = eees[i % len(eees)]
            ideal = round(random.uniform(20.0, 100.0), 2)
            func = "Functional" if random.random() > 0.2 else "Non-Functional"
            schemes_rows.append({
                "id": sid,
                "scheme_name": f"ExecScheme_{sid}",
                "functionality": func,
                "so_name": aee,
                "ideal_per_day": ideal,
                "scheme_label": random.choice(villages) + " PWSS"
            })
            sid += 1

        for jm in jm_list:
            jm_prob = random.uniform(0.10, 0.95)
            for d in range(30):
                if random.random() < jm_prob:
                    date_iso = (today - datetime.timedelta(days=d)).isoformat()
                    time_str = f"{random.randint(6,11)}:{random.choice([0,15,30,45]):02d} AM"
                    scheme_choice = random.choice(schemes_rows)
                    readings_rows.append({
                        "id": rid,
                        "scheme_id": scheme_choice["id"],
                        "jalmitra": jm,
                        "reading": random.choice([110010,215870,150340,189420,200015,234870]),
                        "reading_date": date_iso,
                        "reading_time": time_str,
                        "water_quantity": round(random.uniform(10.0, 100.0), 2),
                        "scheme_name": scheme_choice["scheme_label"],
                        "so_name": scheme_choice["so_name"]
                    })
                    rid += 1

        st.session_state["exec_schemes"] = pd.DataFrame(schemes_rows)
        st.session_state["exec_readings"] = pd.DataFrame(readings_rows)
        st.session_state["exec_demo_generated"] = True
        st.success("✅ Exec demo generated: 300 schemes, 250 jalmitras (AEE Alpha & AEE Beta).")

    st.markdown("---")
    if not st.session_state.get("exec_demo_generated", False):
        st.info("Generate Exec demo to view Executive-level charts (300 schemes, 250 jalmitras).")
    else:
        exec_schemes = st.session_state.get("exec_schemes", pd.DataFrame())
        exec_readings = st.session_state.get("exec_readings", pd.DataFrame())

        total_exec_schemes = len(exec_schemes)
        num_exec_func = int(exec_schemes[exec_schemes["functionality"] == "Functional"]["id"].nunique())
        num_exec_non = max(total_exec_schemes - num_exec_func, 0)

        today_iso = datetime.date.today().isoformat()
        present_exec = int(exec_readings[exec_readings["reading_date"] == today_iso]["jalmitra"].nunique()) if not exec_readings.empty else 0
        absent_exec = max(250 - present_exec, 0)

        c1, c2 = st.columns([1,1])
        with c1:
            st.markdown(f"<small>Functional: <b>{num_exec_func}</b> • Non-Functional: <b>{num_exec_non}</b> • Total: <b>{total_exec_schemes}</b></small>", unsafe_allow_html=True)
            func_counts_exec = pd.Series({"Functional": num_exec_func, "Non-Functional": num_exec_non})
            figf = px.pie(names=func_counts_exec.index, values=func_counts_exec.values, color=func_counts_exec.index,
                          color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
            figf.update_traces(textinfo='percent+label')
            st.plotly_chart(figf, use_container_width=True, height=320)
        with c2:
            st.markdown(f"<small>Present: <b>{present_exec}</b> • Absent: <b>{absent_exec}</b></small>", unsafe_allow_html=True)
            df_exec_part = pd.DataFrame({"status":["Present","Absent"], "count":[present_exec, absent_exec]})
            figp = px.pie(df_exec_part, names="status", values="count", color="status", color_discrete_map={"Present":"#4CAF50","Absent":"#F44336"})
            figp.update_traces(textinfo='percent+label')
            st.plotly_chart(figp, use_container_width=True, height=320)

        st.markdown("---")
        st.subheader("Two AEE Performances (from Exec demo)")
        aees = exec_schemes["so_name"].unique().tolist()
        aee_cards = []
        for a in aees:
            a_schemes = exec_schemes[exec_schemes["so_name"] == a]
            a_read = exec_readings[exec_readings["so_name"] == a] if not exec_readings.empty else pd.DataFrame()
            a_present_jm = int(a_read[a_read["reading_date"] == today_iso]["jalmitra"].nunique()) if not a_read.empty else 0
            a_total_schemes = len(a_schemes)
            a_func = int(a_schemes[a_schemes["functionality"] == "Functional"]["id"].nunique())
            a_non = max(a_total_schemes - a_func, 0)
            aee_cards.append({
                "aee": a,
                "total_schemes": a_total_schemes,
                "functional": a_func,
                "non_functional": a_non,
                "present_jalmitras": a_present_jm
            })

        ac1, ac2 = st.columns(2)
        for i, card in enumerate(aee_cards[:2]):
            col = ac1 if i == 0 else ac2
            with col:
                st.markdown(f"### {card['aee']}")
                st.markdown(f"- Total schemes: **{card['total_schemes']}**")
                st.markdown(f"- Functional: **{card['functional']}**  •  Non-Functional: **{card['non_functional']}**")
                st.markdown(f"- Present Jalmitras (today): **{card['present_jalmitras']}**")
                url = f"?role=Assistant+Executive+Engineer"
                st.markdown(f'<a href="{url}" target="_blank"><button style="padding:8px 12px;border-radius:6px;border:1px solid #2b6; background: linear-gradient(90deg,#b9f0c7,#7ee092);">Open AEE Dashboard</button></a>', unsafe_allow_html=True)

    st.markdown("---")
    st.info("Executive view uses a separate Exec demo dataset (exec_schemes / exec_readings) so it does not modify the main SO/AEE demo data.")

# --------------------------- Section Officer dashboard renderer (preserve SO page) -----------
def render_so_dashboard(so_to_render: str):
    so = so_to_render
    today = datetime.date.today()
    st.header(f"Section Officer Dashboard — {so}")
    st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")

    # Demo generator & remover for SO
    st.markdown("### 🧪 Demo Data Management (SO)")
    colg, colr = st.columns([2,1])
    with colg:
        total_schemes = st.number_input("Total demo schemes (SO)", min_value=4, max_value=150, value=20, key=f"so_total_schemes_{so}")
        if st.button("Generate Demo Data (SO)", key=f"gen_so_{so}"):
            generate_demo_data(int(total_schemes), so_name=so)
    with colr:
        if st.button("Remove Demo Data (SO)", key=f"rem_so_{so}"):
            reset_session_data()
            st.warning("🗑️ All SO demo data removed from session.")

    st.markdown("---")

    schemes_all = st.session_state.get("schemes", pd.DataFrame()).copy()
    readings_all = st.session_state.get("readings", pd.DataFrame()).copy()

    # If no schemes for this SO, instruct to generate
    if schemes_all.empty or schemes_all[schemes_all.get("so_name","")==so].empty:
        st.info("No schemes found for this SO. Use 'Generate Demo Data (SO)' above to create demo data for this SO.")
        return

    schemes = schemes_all[schemes_all["so_name"] == so].copy()
    readings = readings_all[readings_all.get("so_name","") == so].copy()

    # Build master_jalmitras from per-SO scheme_jalmitra_map (fixed)
    scheme_jm_map_for_so = st.session_state.get("scheme_jalmitra_map", {}).get(so, {})
    master_jalmitras = []
    if scheme_jm_map_for_so:
        for sid, jm in scheme_jm_map_for_so.items():
            if sid in schemes["id"].values:
                master_jalmitras.append(jm)
    master_jalmitras = sorted(list(dict.fromkeys(master_jalmitras)))

    # fallback: if master empty, use jalmitras_map or readings
    if not master_jalmitras:
        master_jalmitras = st.session_state.get("jalmitras_map", {}).get(so, [])
    if not master_jalmitras:
        master_jalmitras = sorted(readings["jalmitra"].dropna().unique().tolist())

    func_counts = schemes["functionality"].value_counts()
    today_iso = today.isoformat()

    merged_all = ensure_columns(readings.copy(), ["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name","so_name"]) \
                 .merge(ensure_columns(schemes.copy(), ["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"])[["id","scheme_name","functionality","so_name","ideal_per_day","scheme_label"]],
                        left_on="scheme_id", right_on="id", how="left", suffixes=("_reading","_scheme"))

    # unify so_name and scheme name into consistent columns
    def _col_or_empty(df, colname, fallback=""):
        if colname in df.columns:
            return df[colname]
        else:
            return pd.Series([fallback] * len(df), index=df.index)

    so_reading = _col_or_empty(merged_all, "so_name_reading")
    so_scheme = _col_or_empty(merged_all, "so_name_scheme")
    merged_all["so_name"] = so_reading.fillna("").replace("", np.nan).fillna(so_scheme.fillna("").astype(str)).fillna("").astype(str)

    scheme_name_reading = _col_or_empty(merged_all, "scheme_name_reading")
    scheme_name_scheme = _col_or_empty(merged_all, "scheme_name_scheme")
    merged_all["scheme_name"] = scheme_name_reading.combine_first(scheme_name_scheme).fillna(merged_all.get("scheme_name",""))

    # ensure functionality exists
    merged_all["functionality"] = _col_or_empty(merged_all, "functionality").fillna("")

    today_upd = merged_all[
        (merged_all.get("reading_date", "") == today_iso) &
        (merged_all.get("functionality", "") == "Functional") &
        (merged_all.get("so_name", "") == so)
    ].copy()

    # If demo not generated, instruct and do not fabricate present/absent
    if not st.session_state.get("demo_generated", False):
        st.info("No demo data generated. Generate demo data for this SO (using the button above) or via the AEE demo to populate readings and BFM updates.")
        return

    # compute present jalmitras from today_upd (exact matches)
    present_jalmitras = sorted(today_upd["jalmitra"].dropna().unique().tolist()) if not today_upd.empty else []
    absent_jalmitras = [jm for jm in master_jalmitras if jm not in present_jalmitras]

    present_count = len(present_jalmitras)
    absent_count = len(absent_jalmitras)

    # show pies
    if st.session_state.get("view_mode","Web View") == "Web View":
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Scheme Functionality")
            f_present = int(func_counts.get("Functional", 0))
            f_non = int(func_counts.get("Non-Functional", 0))
            st.markdown(f"<small>Functional: <b>{f_present}</b> • Non-Functional: <b>{f_non}</b></small>", unsafe_allow_html=True)
            fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                          color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
            fig1.update_traces(textinfo='percent+label')
            st.plotly_chart(fig1, use_container_width=True, height=220)
        with c2:
            st.markdown(f"<small>Present: <b>{present_count}</b> &nbsp;&nbsp; Absent: <b>{absent_count}</b></small>", unsafe_allow_html=True)
            st.markdown("#### Jalmitra Updates (Today)")
            df_part = pd.DataFrame({"status":["Present","Absent"], "count":[present_count, absent_count]})
            if df_part["count"].sum() == 0:
                df_part = pd.DataFrame({"status":["Present","Absent"], "count":[0, len(master_jalmitras) if master_jalmitras else 1]})
            fig2 = px.pie(df_part, names='status', values='count', color='status',
                          color_discrete_map={"Present":"#4CAF50","Absent":"#F44336"})
            fig2.update_traces(textinfo='percent+label+value')
            fig2.update_layout(title=f"Present: {present_count} • Absent: {absent_count}", margin=dict(t=30,b=10))
            st.plotly_chart(fig2, use_container_width=True, height=260)
    else:
        st.markdown("#### Scheme Functionality")
        f_present = int(func_counts.get("Functional", 0))
        f_non = int(func_counts.get("Non-Functional", 0))
        st.markdown(f"<small>Functional: <b>{f_present}</b> • Non-Functional: <b>{f_non}</b></small>", unsafe_allow_html=True)
        fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                      color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
        fig1.update_traces(textinfo='percent+label')
        st.plotly_chart(fig1, use_container_width=True, height=220)

        st.markdown(f"<small>Present: <b>{present_count}</b> &nbsp;&nbsp; Absent: <b>{absent_count}</b></small>", unsafe_allow_html=True)
        st.markdown("#### Jalmitra Updates (Today)")
        df_part = pd.DataFrame({"status":["Present","Absent"], "count":[present_count, absent_count]})
        if df_part["count"].sum() == 0:
            df_part = pd.DataFrame({"status":["Present","Absent"], "count":[0, len(master_jalmitras) if master_jalmitras else 1]})
        fig2 = px.pie(df_part, names='status', values='count', color='status',
                      color_discrete_map={"Present":"#4CAF50","Absent":"#F44336"})
        fig2.update_traces(textinfo='percent+label+value')
        fig2.update_layout(title=f"Present: {present_count} • Absent: {absent_count}", margin=dict(t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True, height=240)

    st.markdown("---")

    # BFM table (unchanged)
    st.subheader("🧾 BFM Readings Updated Today")
    bfm_df = merged_all[
        (merged_all.get("reading_date", "") == today_iso) &
        (merged_all.get("functionality", "") == "Functional") &
        (merged_all.get("so_name", "") == so)
    ].copy()
    if not bfm_df.empty:
        display_bfm = bfm_df[["jalmitra", "scheme_name", "reading", "reading_time", "water_quantity"]].copy()
        display_bfm.insert(0, "S.No", range(1, len(display_bfm)+1))
        display_bfm = display_bfm.rename(columns={"reading":"BFM Reading", "reading_time":"Reading Time", "water_quantity":"Water Quantity (m³)", "jalmitra":"Jalmitra", "scheme_name":"Scheme Name"})
        try:
            def _highlight_first(row):
                return ['background-color: #e6ffed' if row.name == 0 else '' for _ in row]
            sty = display_bfm.style.format({"Water Quantity (m³)":"{:.2f}"}).apply(_highlight_first, axis=1)
            st.dataframe(sty, height=300, use_container_width=True)
        except Exception:
            st.table(display_bfm)
    else:
        st.info("No BFM readings updated today for this SO.")

    st.markdown("---")

    # ------------------ Restored Section: Jalmitra Performance (Top & Bottom), Absent list, Tap names & charts ------------------
    st.subheader("🏅 Jalmitra Performance — Top & Bottom (split full list)")
    period = st.selectbox("Show performance for", [7, 15, 30], index=0, format_func=lambda x: f"{x} days", key=f"so_period_{so}")
    start_date = (today - datetime.timedelta(days=period-1)).isoformat()
    end_date = today_iso

    lastN, metrics = compute_metrics(readings_all, schemes_all, so, start_date, end_date)

    if lastN.empty or metrics.empty:
        st.info(f"No readings in the last {period} days for this SO.")
    else:
        # ensure master jalmitras included
        for jm in master_jalmitras:
            if jm not in metrics["jalmitra"].values:
                metrics = pd.concat([metrics, pd.DataFrame([{
                    "jalmitra": jm, "days_updated": 0, "total_water_m3": 0.0, "schemes_covered": 0, "ideal_total_Nd": 0.0, "quantity_score": 0.0
                }])], ignore_index=True)

        metrics["days_norm"] = metrics["days_updated"] / float(period)
        metrics["score"] = (0.5 * metrics["days_norm"]) + (0.5 * metrics["quantity_score"])
        metrics = metrics.sort_values(by=["score","total_water_m3"], ascending=False).reset_index(drop=True)
        metrics["Rank"] = metrics.index + 1

        villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
        rnd = random.Random(42)
        metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]
        metrics["ideal_total_Nd"] = metrics.get("ideal_total_Nd", 0.0).round(2)

        total_jm_count = len(metrics)
        half = total_jm_count // 2
        top_count = half
        bottom_count = total_jm_count - top_count

        top_metrics = metrics.head(top_count).copy()
        bottom_metrics = metrics.tail(bottom_count).sort_values(by=["score","total_water_m3"], ascending=True).reset_index(drop=True).copy()

        top_table = top_metrics[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
        top_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

        bottom_table = bottom_metrics[["Rank","jalmitra","Scheme Name","days_updated","total_water_m3","ideal_total_Nd","score"]].copy()
        bottom_table.columns = ["Rank","Jalmitra","Scheme Name",f"Days Updated (last {period}d)","Total Water (m³)","Ideal Water (m³)","Score"]

        def styled_df(df, cmap):
            sty = df.style.format({"Total Water (m³)":"{:.2f}","Ideal Water (m³)":"{:.2f}","Score":"{:.3f}"})
            sty = sty.background_gradient(subset=[f"Days Updated (last {period}d)","Total Water (m³)","Score"], cmap=cmap)
            sty = sty.set_table_styles([
                {"selector":"th","props":[("font-weight","600"),("border","1px solid #ddd"),("background-color","#f7f7f7")]},
                {"selector":"td","props":[("border","1px solid #eee")]}
            ])
            return sty

        col_t, col_w = st.columns([1,1])
        with col_t:
            st.markdown(f"### 🟢 Top performing — last {period} days")
            st.dataframe(styled_df(top_table, "Greens"), height=360)
            st.download_button(f"⬇️ Download Top — {so} (CSV)", top_table.to_csv(index=False).encode("utf-8"), f"top_{so}.csv")
        with col_w:
            st.markdown(f"### 🔴 Bottom performing — last {period} days")
            st.dataframe(styled_df(bottom_table, "Reds_r"), height=360)
            st.download_button(f"⬇️ Download Bottom — {so} (CSV)", bottom_table.to_csv(index=False).encode("utf-8"), f"bottom_{so}.csv")

        # Absent Jalmitras and assigned scheme (one-to-one)
        absent_info = []
        jm_scheme_map = st.session_state.get("jalmitra_scheme_map", {}).get(so, {})
        scheme_jm_map = st.session_state.get("scheme_jalmitra_map", {}).get(so, {})
        for jm in absent_jalmitras:
            assigned_label = jm_scheme_map.get(jm)
            if not assigned_label:
                found = [sid for sid, ajm in scheme_jm_map.items() if ajm == jm and sid in schemes["id"].values]
                if found:
                    assigned_label = schemes[schemes["id"].isin(found)]["scheme_label"].unique().tolist()[0]
            schemes_text = assigned_label if assigned_label else "—"
            absent_info.append({"Jalmitra": jm, "Assigned Scheme": schemes_text})

        st.markdown("---")
        st.markdown(f"**Absent Jalmitras (today: {today_iso}) — {len(absent_info)}**")
        if absent_info:
            absent_df = pd.DataFrame(absent_info)
            try:
                sty = absent_df.style.set_table_styles([{"selector":"th","props":[("font-weight","600"),("background-color","#fff1f0")]},
                                                       {"selector":"td","props":[("border","1px solid #eee")]}])
                st.dataframe(sty, height=220)
            except Exception:
                st.table(absent_df)
            st.download_button("⬇️ Download Absent Jalmitras (today) CSV", absent_df.to_csv(index=False).encode("utf-8"), f"absent_jalmitras_{today_iso}_{so}.csv")
        else:
            st.info("No absent Jalmitras today (all updated).")

        # clickable names to show period chart inline
        st.markdown("**Tap a name below to open the performance chart**")
        top_names = top_table["Jalmitra"].tolist()
        bottom_names = bottom_table["Jalmitra"].tolist()

        if st.session_state.get("view_mode","Web View") == "Web View":
            with st.container():
                st.markdown("**Top — Tap name**")
                if top_names:
                    cols = st.columns(len(top_names))
                    for i, name in enumerate(top_names):
                        if cols[i].button(name, key=f"btn_top_{so}_{period}_{i}_{name}"):
                            st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name
            with st.container():
                st.markdown("**Bottom — Tap name**")
                if bottom_names:
                    cols = st.columns(len(bottom_names))
                    for i, name in enumerate(bottom_names):
                        if cols[i].button(name, key=f"btn_bottom_{so}_{period}_{i}_{name}"):
                            st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name
        else:
            st.markdown("**Top — Tap a name**")
            for i, name in enumerate(top_names):
                if st.button(name, key=f"pbtn_top_{so}_{period}_{i}_{name}"):
                    st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name
            st.markdown("**Bottom — Tap a name**")
            for i, name in enumerate(bottom_names):
                if st.button(name, key=f"pbtn_bottom_{so}_{period}_{i}_{name}"):
                    st.session_state["selected_jalmitra"] = None if st.session_state.get("selected_jalmitra") == name else name

        # Inline performance chart for selected jalmitra (only if belongs to this SO)
        sel_jm = st.session_state.get("selected_jalmitra")
        if sel_jm and sel_jm in metrics["jalmitra"].values:
            st.markdown("---")
            st.subheader(f"Performance — {sel_jm}")
            last_window, _ = compute_metrics(readings_all, schemes_all, so, start_date, end_date)
            jm_data = last_window[last_window["jalmitra"] == sel_jm] if (not last_window.empty) else pd.DataFrame()
            dates = [(today - datetime.timedelta(days=d)).isoformat() for d in reversed(range(period))]
            if jm_data.empty:
                st.info("No readings for this Jalmitra in the selected window.")
                daily = pd.DataFrame({"reading_date": dates, "water_quantity": [0.0]*len(dates)})
                ideal_val = 0.0
            else:
                daily = jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates, fill_value=0).reset_index()
                daily.columns = ["reading_date","water_quantity"]
                daily["water_quantity"] = daily["water_quantity"].round(2)
                jm_scheme_label = st.session_state.get("jalmitra_scheme_map", {}).get(so, {}).get(sel_jm)
                if not jm_scheme_label:
                    reverse_map = {v:k for k,v in st.session_state.get("scheme_jalmitra_map", {}).get(so, {}).items()}
                    sid = reverse_map.get(sel_jm)
                    if sid:
                        matched = schemes_all[schemes_all["id"]==sid]
                        ideal_val = matched["ideal_per_day"].iloc[0] if not matched.empty else 0.0
                    else:
                        ideal_val = 0.0
                else:
                    matched = schemes_all[schemes_all["scheme_label"] == jm_scheme_label]
                    ideal_val = matched["ideal_per_day"].iloc[0] if not matched.empty else 0.0
                ideal_total = round(float(ideal_val) * period, 2)
            per_day_ideal = ideal_val
            daily["color_flag"] = daily["water_quantity"].apply(lambda x: "above" if x >= per_day_ideal and per_day_ideal>0 else "below")
            fig = px.bar(daily, x="reading_date", y="water_quantity", labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                         title=f"{sel_jm} — Daily Water Supplied (Last {period} Days)",
                         color="color_flag", color_discrete_map={"above":"#2e7d32","below":"#c62828"})
            if per_day_ideal > 0:
                fig.add_hline(y=per_day_ideal, line_dash="dash", line_color="red", annotation_text=f"Ideal/day: {per_day_ideal:.2f} m³", annotation_position="top left")
            fig.update_layout(showlegend=False, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True, height=380)

            download_df = daily.rename(columns={"reading_date":"Date","water_quantity":"Water (m3)"}).copy()
            download_df["Ideal per day (m3)"] = per_day_ideal
            download_df["Ideal total (m3)"] = per_day_ideal * period
            st.markdown(f"**Total ({period} days):** {download_df['Water (m3)'].sum():.2f} m³  **Days Updated:** {(download_df['Water (m3)']>0).sum()}/{period}")
            st.download_button(f"⬇️ Download {sel_jm} readings (last {period} days)", download_df.to_csv(index=False).encode("utf-8"), file_name=f"{sel_jm}_readings_{period}d.csv")

            if st.session_state.get("view_mode","Web View") == "Web View":
                if st.button("Close View"):
                    st.session_state["selected_jalmitra"] = None
            else:
                if st.button("Close View (Phone)"):
                    st.session_state["selected_jalmitra"] = None

# --------------------------- Render logic -----------
if role == "Section Officer":
    query_params = st.query_params
    query_so = query_params.get("so", None)

    if query_so:
        render_so_dashboard(query_so)
    else:
        render_so_dashboard("ROKI RAY")

elif role == "Assistant Executive Engineer":
    # AEE page already rendered above when role selected
    pass
elif role == "Executive Engineer":
    # Exec page rendered above
    pass
else:
    st.header(f"{role} Dashboard — Placeholder")
    st.info("Placeholder view. Implement similarly when needed.")

# --------------------------- Exports & footer -----------
st.markdown("---")
st.subheader("📤 Export Snapshot")
schemes_df = st.session_state.get("schemes", pd.DataFrame())
readings_df = st.session_state.get("readings", pd.DataFrame())
st.download_button("Schemes CSV", schemes_df.to_csv(index=False).encode("utf-8"), "schemes.csv")
st.download_button("Readings CSV", readings_df.to_csv(index=False).encode("utf-8"), "readings.csv")
st.success(f"Dashboard ready. Demo data generated: {st.session_state.get('demo_generated', False)}")
