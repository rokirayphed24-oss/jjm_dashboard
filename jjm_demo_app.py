# jjm_demo_app.py
# JJM Dashboard — Names-in-table are clickable (Top & Worst)
# - Top/Worst tables rendered as interactive rows where NAME is a button
# - Coloring: greens for best, reds for worst (intensity scaled)
# - All previous features preserved (BFM table with time, S.No start=1, demo data caps, Web/Phone view)

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px

# --------------------------- Page setup ---------------------------
st.set_page_config(page_title="JJM Dashboard — Names Clickable", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("For Section Officer **ROKI RAY** — Tap a name to view 7-day performance.")
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
    st.session_state.setdefault("next_scheme_id", 1)
    st.session_state.setdefault("next_reading_id", 1)
    st.session_state.setdefault("demo_generated", False)
    st.session_state.setdefault("selected_jalmitra", None)

init_state()

# color helpers
def green_hex_for_value(v: float) -> str:
    # v expected 0..1 -> map to light->dark green
    v = max(0.0, min(1.0, v))
    # produce hex by mixing with white
    r = int(240 - v * 160)   # from 240 -> 80
    g = int(255 - (1-v) * 80)  # ensure green heavy
    b = int(240 - v * 160)
    return f"#{r:02x}{g:02x}{b:02x}"

def red_hex_for_value(v: float) -> str:
    # v expected 0..1 where 1 = worst (darkest red)
    v = max(0.0, min(1.0, v))
    r = int(255 - (1-v)*50)   # keep red strong
    g = int(240 - v * 200)    # reduce green for darker
    b = int(240 - v * 200)
    return f"#{r:02x}{g:02x}{b:02x}"

# --------------------------- Demo data functions ---------------------------
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

    # create readings for functional schemes only; cap water_quantity to 100 and round to 2 decimals
    for i, s in enumerate(schemes):
        if s["functionality"] != "Functional":
            continue
        scheme_label = random.choice(villages) + " PWSS"
        jalmitra = jalmitras[i % len(jalmitras)]
        for d in range(7):
            date_iso = (today - datetime.timedelta(days=d)).isoformat()
            if random.random() < FIXED_UPDATE_PROB:
                water_qty = round(random.uniform(10.0, 100.0), 2)  # cap at 100
                readings.append({
                    "id": len(readings) + 1,
                    "scheme_id": s["id"],
                    "jalmitra": jalmitra,
                    "reading": random.choice(reading_samples),
                    "reading_date": date_iso,
                    "reading_time": f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
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

# --------------------------- UI: Demo data controls ---------------------------
st.markdown("### 🧪 Demo Data Management")
c1, c2 = st.columns([2,1])
with c1:
    total_schemes = st.number_input("Total demo schemes", min_value=4, max_value=150, value=20)
    if st.button("Generate Demo Data"):
        generate_demo_data(int(total_schemes))
with c2:
    if st.button("Remove Demo Data"):
        reset_session_data(); st.warning("🗑️ All data removed.")
st.markdown("---")

# --------------------------- Main dashboard header ---------------------------
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

# --------------------------- Overview pies ---------------------------
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

# compose scheme display for today rows
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
        dfp = pd.DataFrame({"status":["Updated","Absent"],"count":[updated_count, absent_count]})
        if dfp["count"].sum() == 0:
            dfp = pd.DataFrame({"status":["Updated","Absent"],"count":[0,total_functional if total_functional>0 else 1]})
        fig2 = px.pie(dfp, names="status", values="count", color="status", color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
        fig2.update_traces(textinfo='percent+label')
        st.plotly_chart(fig2, use_container_width=True, height=220)
else:
    st.markdown("#### Scheme Functionality")
    fig1 = px.pie(names=func_counts.index, values=func_counts.values, color=func_counts.index,
                  color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"})
    fig1.update_traces(textinfo='percent+label')
    st.plotly_chart(fig1, use_container_width=True, height=240)
    st.markdown("#### Jalmitra Updates (Today)")
    dfp = pd.DataFrame({"status":["Updated","Absent"],"count":[updated_count, absent_count]})
    if dfp["count"].sum() == 0:
        dfp = pd.DataFrame({"status":["Updated","Absent"],"count":[0,total_functional if total_functional>0 else 1]})
    fig2 = px.pie(dfp, names="status", values="count", color="status", color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"})
    fig2.update_traces(textinfo='percent+label')
    st.plotly_chart(fig2, use_container_width=True, height=240)

st.markdown("---")

# --------------------------- Compute rankings ---------------------------
start_date = (today - datetime.timedelta(days=6)).isoformat()
end_date = today_iso
last7, metrics = compute_metrics(readings, schemes, so, start_date, end_date)

if last7.empty:
    st.info("No readings in last 7 days.")
else:
    max_total = metrics["total_water_m3"].max() if not metrics["total_water_m3"].empty else 0.0
    metrics["score"] = 0.5 * (metrics["days_updated"]/7.0) + (0.5 * (metrics["total_water_m3"]/max_total) if max_total>0 else 0.0)
    metrics = metrics.sort_values(by=["score","total_water_m3"], ascending=False).reset_index(drop=True)
    metrics["Rank"] = metrics.index + 1
    # deterministic illustrative Scheme Name
    villages = ["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    rnd = random.Random(42)
    metrics["Scheme Name"] = [rnd.choice(villages) + " PWSS" for _ in range(len(metrics))]

    top10 = metrics.head(10).copy()
    worst10 = metrics.sort_values(by="score", ascending=True).head(10).copy()

    # normalize score 0..1 for coloring
    if not top10.empty:
        top_min, top_max = top10["score"].min(), top10["score"].max()
        top_range = top_max - top_min if top_max != top_min else 1.0
        top10["score_norm"] = (top10["score"] - top_min) / top_range
    else:
        top10["score_norm"] = 0.0
    if not worst10.empty:
        w_min, w_max = worst10["score"].min(), worst10["score"].max()
        w_range = w_max - w_min if w_max != w_min else 1.0
        # for worst coloring we want 1.0 = worst (lowest score) -> invert mapping
        worst10["score_norm_inv"] = 1.0 - ((worst10["score"] - w_min) / w_range)
    else:
        worst10["score_norm_inv"] = 0.0

    # --------------------------- Render interactive Top/Worst tables (names clickable) ---------------------------
    st.markdown("### 🏅 Jalmitra Performance — Click name to toggle 7-day chart")
    if view_mode == "Web View":
        col_left, col_right = st.columns([1,1])
        with col_left:
            st.markdown("#### 🟢 Top 10 Performing Jalmitras")
            # header
            hdr_cols = st.columns([0.5,2.2,2.2,1.2,1.4,1.0])
            hdr_cols[0].markdown("**#**")
            hdr_cols[1].markdown("**Jalmitra**")
            hdr_cols[2].markdown("**Scheme Name**")
            hdr_cols[3].markdown("**Days (7d)**")
            hdr_cols[4].markdown("**Total Water (m³)**")
            hdr_cols[5].markdown("**Score**")
            # rows
            for i, r in top10.reset_index(drop=True).iterrows():
                cols = st.columns([0.5,2.2,2.2,1.2,1.4,1.0])
                cols[0].write(int(r["Rank"]))
                # name button (toggle)
                if cols[1].button(str(r["jalmitra"]), key=f"top_name_{i}_{r['jalmitra']}"):
                    if st.session_state.get("selected_jalmitra") == r["jalmitra"]:
                        st.session_state["selected_jalmitra"] = None
                    else:
                        st.session_state["selected_jalmitra"] = r["jalmitra"]
                cols[2].write(r["Scheme Name"])
                cols[3].markdown(f"<div style='background:{green_hex_for_value(r['score_norm'])};padding:6px;border-radius:6px;text-align:center;color:#000;font-weight:600'>{int(r['days_updated'])}</div>", unsafe_allow_html=True)
                cols[4].markdown(f"<div style='background:{green_hex_for_value(r['score_norm'])};padding:6px;border-radius:6px;text-align:right;color:#000;font-weight:600'>{r['total_water_m3']:.2f}</div>", unsafe_allow_html=True)
                cols[5].markdown(f"<div style='background:{green_hex_for_value(r['score_norm'])};padding:6px;border-radius:6px;text-align:center;color:#000;font-weight:600'>{r['score']:.3f}</div>", unsafe_allow_html=True)

        with col_right:
            st.markdown("#### 🔴 Worst 10 Performing Jalmitras")
            hdr_cols = st.columns([0.5,2.2,2.2,1.2,1.4,1.0])
            hdr_cols[0].markdown("**#**")
            hdr_cols[1].markdown("**Jalmitra**")
            hdr_cols[2].markdown("**Scheme Name**")
            hdr_cols[3].markdown("**Days (7d)**")
            hdr_cols[4].markdown("**Total Water (m³)**")
            hdr_cols[5].markdown("**Score**")
            for i, r in worst10.reset_index(drop=True).iterrows():
                cols = st.columns([0.5,2.2,2.2,1.2,1.4,1.0])
                cols[0].write(int(r["Rank"]))
                if cols[1].button(str(r["jalmitra"]), key=f"worst_name_{i}_{r['jalmitra']}"):
                    if st.session_state.get("selected_jalmitra") == r["jalmitra"]:
                        st.session_state["selected_jalmitra"] = None
                    else:
                        st.session_state["selected_jalmitra"] = r["jalmitra"]
                cols[2].write(r["Scheme Name"])
                # compute normalized inv for this row (use earlier column if present)
                inv = r.get("score_norm_inv", 0.5)
                cols[3].markdown(f"<div style='background:{red_hex_for_value(inv)};padding:6px;border-radius:6px;text-align:center;color:#000;font-weight:600'>{int(r['days_updated'])}</div>", unsafe_allow_html=True)
                cols[4].markdown(f"<div style='background:{red_hex_for_value(inv)};padding:6px;border-radius:6px;text-align:right;color:#000;font-weight:600'>{r['total_water_m3']:.2f}</div>", unsafe_allow_html=True)
                cols[5].markdown(f"<div style='background:{red_hex_for_value(inv)};padding:6px;border-radius:6px;text-align:center;color:#000;font-weight:600'>{r['score']:.3f}</div>", unsafe_allow_html=True)

    else:
        # Phone view stacked: top then worst
        st.markdown("#### 🟢 Top 10 Performing Jalmitras")
        for i, r in top10.reset_index(drop=True).iterrows():
            cols = st.columns([0.8,2.8,1.6])
            cols[0].write(int(r["Rank"]))
            if cols[1].button(str(r["jalmitra"]), key=f"p_top_name_{i}_{r['jalmitra']}"):
                if st.session_state.get("selected_jalmitra") == r["jalmitra"]:
                    st.session_state["selected_jalmitra"] = None
                else:
                    st.session_state["selected_jalmitra"] = r["jalmitra"]
            cols[2].markdown(f"<div style='padding:6px;border-radius:6px;background:{green_hex_for_value(r['score_norm'])};text-align:center;font-weight:600'>{r['score']:.3f}</div>", unsafe_allow_html=True)
            st.write(f"• {r['Scheme Name']} — Days: {int(r['days_updated'])} — Water: {r['total_water_m3']:.2f} m³")
        st.markdown("#### 🔴 Worst 10 Performing Jalmitras")
        for i, r in worst10.reset_index(drop=True).iterrows():
            cols = st.columns([0.8,2.8,1.6])
            cols[0].write(int(r["Rank"]))
            if cols[1].button(str(r["jalmitra"]), key=f"p_worst_name_{i}_{r['jalmitra']}"):
                if st.session_state.get("selected_jalmitra") == r["jalmitra"]:
                    st.session_state["selected_jalmitra"] = None
                else:
                    st.session_state["selected_jalmitra"] = r["jalmitra"]
            inv = r.get("score_norm_inv", 0.5)
            cols[2].markdown(f"<div style='padding:6px;border-radius:6px;background:{red_hex_for_value(inv)};text-align:center;font-weight:600'>{r['score']:.3f}</div>", unsafe_allow_html=True)
            st.write(f"• {r['Scheme Name']} — Days: {int(r['days_updated'])} — Water: {r['total_water_m3']:.2f} m³")

    # downloads
    st.download_button("⬇️ Download Top 10 CSV", top10.to_csv(index=False).encode("utf-8"), "top_10_jalmitras.csv")
    st.download_button("⬇️ Download Worst 10 CSV", worst10.to_csv(index=False).encode("utf-8"), "worst_10_jalmitras.csv")

# --------------------------- Show 7-day chart directly under the tables when a name is clicked ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm = st.session_state["selected_jalmitra"]
    st.markdown("---")
    st.subheader(f"7-day Performance — {jm}")
    if 'last7' not in locals() or last7 is None:
        last7, _ = compute_metrics(readings, schemes, so, start_date, end_date) if ('start_date' in locals() and 'end_date' in locals()) else compute_metrics(readings, schemes, so, (today - datetime.timedelta(days=6)).isoformat(), today_iso)
    jm_data = last7[last7["jalmitra"] == jm] if (not last7.empty) else pd.DataFrame()
    if jm_data.empty:
        st.info("No readings for this Jalmitra in the last 7 days.")
    else:
        dates = [(today - datetime.timedelta(days=d)).isoformat() for d in reversed(range(7))]
        daily = jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates, fill_value=0).reset_index()
        daily["water_quantity"] = daily["water_quantity"].round(2)
        fig = px.bar(daily, x="reading_date", y="water_quantity",
                     labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                     title=f"{jm} — Daily Water Supplied (Last 7 Days)")
        st.plotly_chart(fig, use_container_width=True, height=380)
        st.markdown(f"**Total (7 days):** {daily['water_quantity'].sum():.2f} m³  **Days Updated:** {(daily['water_quantity']>0).sum()}/7")
    if view_mode == "Web View":
        if st.button("Close 7-day View"):
            st.session_state["selected_jalmitra"] = None
    else:
        if st.button("Close 7-day View (Phone)"):
            st.session_state["selected_jalmitra"] = None

st.markdown("---")

# --------------------------- BFM Readings Updated Today (with reading_time) ---------------------------
st.subheader("📅 BFM Readings Updated Today")
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    cols_needed = ["jalmitra","reading","water_quantity","reading_time","scheme_name_reading","scheme_name_scheme","Scheme Display"]
    today_upd = ensure_columns(today_upd.copy(), cols_needed)
    today_upd["reading_time_norm"] = today_upd["reading_time"].astype(str).replace("", "00:00:00")
    today_upd_sorted = today_upd.sort_values(by=["jalmitra","reading_time_norm"], ascending=[True, False])
    latest_per_jm = today_upd_sorted.drop_duplicates(subset=["jalmitra"], keep="first").copy()

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
    latest_per_jm["reading"] = pd.to_numeric(latest_per_jm["reading"], errors="coerce").fillna(0).astype(int)
    latest_per_jm["water_quantity"] = pd.to_numeric(latest_per_jm["water_quantity"], errors="coerce").fillna(0.0).round(2)
    latest_per_jm["BFM Reading Display"] = latest_per_jm["reading"].apply(lambda x: f"{int(x):06d}")

    daily_bfm = latest_per_jm[["jalmitra","Scheme Name","BFM Reading Display","water_quantity","reading_time"]].copy()
    daily_bfm.columns = ["Jalmitra","Scheme Name","BFM Reading","Water Quantity (m³)","Reading Time"]
    daily_bfm = daily_bfm.sort_values("Jalmitra").reset_index(drop=True)
    daily_bfm.insert(0, "S.No", range(1, len(daily_bfm)+1))

    # show table (styled)
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
