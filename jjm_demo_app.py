# jjm_demo_app.py
# JJM Dashboard — Full version:
# Styled Top/Worst tables, clickable “View” charts, and “📅 BFM Readings Updated Today” table.

import streamlit as st
import pandas as pd
import datetime
import random
import plotly.express as px
from typing import Tuple

# ---------------------------  Page setup  ---------------------------
st.set_page_config(page_title="JJM Dashboard — Full", layout="wide")
st.title("Jal Jeevan Mission — Unified Dashboard")
st.markdown("For Section Officer **ROKI RAY** — Jalmitra performance, daily updates, and readings.")
st.markdown("---")

# ---------------------------  Helpers  ---------------------------
def ensure_columns(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    if df is None:
        df = pd.DataFrame(columns=cols)
    for c in cols:
        if c not in df.columns:
            df[c] = 0 if c in ("id", "scheme_id", "reading") else 0.0 if c == "water_quantity" else ""
    return df

def init_state():
    for k, v in {
        "schemes": pd.DataFrame(columns=["id","scheme_name","functionality","so_name"]),
        "readings": pd.DataFrame(columns=[
            "id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"]),
        "jalmitras": [], "next_scheme_id": 1, "next_reading_id": 1,
        "demo_generated": False, "selected_jalmitra": None
    }.items(): st.session_state.setdefault(k, v)
init_state()

# ---------------------------  Demo data generator  ---------------------------
def reset_session_data():
    for key in ["schemes","readings","jalmitras"]: st.session_state[key] = [] if key=="jalmitras" else pd.DataFrame()
    st.session_state.update({"next_scheme_id":1,"next_reading_id":1,"demo_generated":False,"selected_jalmitra":None})

def generate_demo_data(total_schemes:int=20, so_name:str="ROKI RAY"):
    FIXED_UPDATE_PROB=0.85
    assamese=["Biren","Nagen","Rahul","Vikram","Debojit","Anup","Kamal","Ranjit","Himangshu",
              "Pranjal","Rupam","Dilip","Utpal","Amit","Jayanta","Hemanta","Rituraj","Dipankar",
              "Bikash","Dhruba","Subham","Pritam","Saurav","Bijoy","Manoj"]
    jalmitras=random.sample(assamese*3,total_schemes)
    villages=["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar",
              "Jorhat","Hajo","Tihu","Kokrajhar","Nalbari","Barpeta","Rangia","Goalpara","Dhemaji",
              "Dibrugarh","Mariani","Sonari"]
    today=datetime.date.today(); readings=[]
    schemes=[{"id":i+1,"scheme_name":f"Scheme {chr(65+i)}",
              "functionality":random.choice(["Functional","Non-Functional"]),
              "so_name":so_name} for i in range(total_schemes)]
    for i,s in enumerate(schemes):
        if s["functionality"]!="Functional": continue
        jm=jalmitras[i%len(jalmitras)]; label=random.choice(villages)+" PWSS"
        for d in range(7):
            if random.random()<FIXED_UPDATE_PROB:
                readings.append({
                    "id":len(readings)+1,"scheme_id":s["id"],"jalmitra":jm,
                    "reading":random.choice([110010,215870,150340,189420,200015,234870]),
                    "reading_date":(today-datetime.timedelta(days=d)).isoformat(),
                    "reading_time":f"{random.randint(6,18)}:{random.choice(['00','15','30','45'])}:00",
                    "water_quantity":round(random.uniform(40,350),2),"scheme_name":label})
    st.session_state.update({
        "schemes":pd.DataFrame(schemes),"readings":pd.DataFrame(readings),
        "jalmitras":jalmitras,"demo_generated":True})
    st.success("✅ Demo data generated for ROKI RAY.")

# ---------------------------  Compute metrics  ---------------------------
@st.cache_data
def compute_metrics(readings,schemes,so,start,end):
    r=ensure_columns(readings.copy(),["id","scheme_id","jalmitra","reading","reading_date","reading_time","water_quantity","scheme_name"])
    s=ensure_columns(schemes.copy(),["id","scheme_name","functionality","so_name"])
    m=r.merge(s[["id","scheme_name","functionality","so_name"]],left_on="scheme_id",right_on="id",how="left")
    mask=(m["functionality"]=="Functional")&(m["so_name"]==so)&(m["reading_date"]>=start)&(m["reading_date"]<=end)
    last7=m.loc[mask]
    if last7.empty: return last7,pd.DataFrame()
    metrics=(last7.groupby("jalmitra")
             .agg(days_updated=("reading_date",lambda x:x.nunique()),total_water_m3=("water_quantity","sum"))
             .reset_index())
    return last7,metrics

# ---------------------------  Demo data UI  ---------------------------
st.markdown("### 🧪 Demo Data Management")
c1,c2=st.columns([2,1])
with c1:
    n=st.number_input("Total demo schemes",4,100,20)
    if st.button("Generate Demo Data"): generate_demo_data(int(n))
with c2:
    if st.button("Remove Demo Data"): reset_session_data(); st.warning("🗑️ All data removed.")
st.markdown("---")

# ---------------------------  Dashboard  ---------------------------
role=st.selectbox("Select Role",["Section Officer","Assistant Executive Engineer","Executive Engineer"])
if role!="Section Officer": st.stop()

so="ROKI RAY"; today=datetime.date.today()
st.header(f"Section Officer Dashboard — {so}")
st.markdown(f"**DATE:** {today.strftime('%A, %d %B %Y').upper()}")

schemes,readings=st.session_state["schemes"],st.session_state["readings"]
if schemes.empty: st.info("Generate demo data first."); st.stop()

# ---------------------------  Pies  ---------------------------
func=schemes["functionality"].value_counts()
today_iso=today.isoformat()
merged=readings.merge(schemes[["id","scheme_name","functionality","so_name"]],
                      left_on="scheme_id",right_on="id",how="left")
today_upd=merged[(merged["reading_date"]==today_iso)&
                 (merged["functionality"]=="Functional")&(merged["so_name"]==so)]
upd=len(today_upd["jalmitra"].unique())
func_total=len(schemes[schemes["functionality"]=="Functional"])
absent=max(func_total-upd,0)
c1,c2=st.columns(2)
with c1:
    st.markdown("#### Scheme Functionality")
    st.plotly_chart(px.pie(names=func.index,values=func.values,
        color=func.index,color_discrete_map={"Functional":"#4CAF50","Non-Functional":"#F44336"}),use_container_width=True)
with c2:
    st.markdown("#### Jalmitra Updates (Today)")
    dfp=pd.DataFrame({"status":["Updated","Absent"],"count":[upd,absent]})
    st.plotly_chart(px.pie(dfp,names="status",values="count",
        color="status",color_discrete_map={"Updated":"#4CAF50","Absent":"#F44336"}),use_container_width=True)

st.markdown("---")

# ---------------------------  Top/Worst + View  ---------------------------
start=(today-datetime.timedelta(days=6)).isoformat(); end=today_iso
last7,metrics=compute_metrics(readings,schemes,so,start,end)
if last7.empty: st.info("No readings in last 7 days.")
else:
    metrics["score"]=0.5*(metrics["days_updated"]/7)+0.5*(metrics["total_water_m3"]/metrics["total_water_m3"].max())
    metrics=metrics.sort_values("score",ascending=False).reset_index(drop=True); metrics["Rank"]=metrics.index+1
    villages=["Rampur","Kahikuchi","Dalgaon","Guwahati","Boko","Moran","Tezpur","Sibsagar","Jorhat","Hajo"]
    metrics["Scheme Name"]=[random.choice(villages)+" PWSS" for _ in range(len(metrics))]
    top10=metrics.head(10).copy(); worst10=metrics.tail(10).sort_values("score")

    c1,c2=st.columns(2)
    with c1:
        st.markdown("### 🟢 Top 10 Jalmitras")
        st.dataframe(top10.style.background_gradient(cmap="Greens",subset=["days_updated","total_water_m3","score"]))
    with c2:
        st.markdown("### 🔴 Worst 10 Jalmitras")
        st.dataframe(worst10.style.background_gradient(cmap="Reds_r",subset=["days_updated","total_water_m3","score"]))

    st.markdown("**Tap View to see 7-day chart**")
    c1,c2=st.columns(2)
    with c1:
        for i,r in top10.iterrows():
            cols=st.columns([0.5,1.5,2,1,1,1,0.8])
            cols[0].write(r["Rank"]); cols[1].write(r["jalmitra"]); cols[2].write(r["Scheme Name"])
            cols[3].write(r["days_updated"]); cols[4].write(f"{r['total_water_m3']:.1f}"); cols[5].write(f"{r['score']:.3f}")
            if cols[6].button("View",key=f"view_top_{i}"): st.session_state["selected_jalmitra"]=r["jalmitra"]
    with c2:
        for i,r in worst10.iterrows():
            cols=st.columns([0.5,1.5,2,1,1,1,0.8])
            cols[0].write(r["Rank"]); cols[1].write(r["jalmitra"]); cols[2].write(r["Scheme Name"])
            cols[3].write(r["days_updated"]); cols[4].write(f"{r['total_water_m3']:.1f}"); cols[5].write(f"{r['score']:.3f}")
            if cols[6].button("View",key=f"view_worst_{i}"): st.session_state["selected_jalmitra"]=r["jalmitra"]

# ---------------------------  View chart  ---------------------------
if st.session_state.get("selected_jalmitra"):
    jm=st.session_state["selected_jalmitra"]; st.markdown("---"); st.subheader(f"7-day Performance — {jm}")
    jm_data=last7[last7["jalmitra"]==jm]
    if jm_data.empty: st.info("No readings for this Jalmitra.")
    else:
        dates=[(today-datetime.timedelta(days=d)).isoformat() for d in reversed(range(7))]
        daily=jm_data.groupby("reading_date")["water_quantity"].sum().reindex(dates,fill_value=0).reset_index()
        fig=px.bar(daily,x="reading_date",y="water_quantity",
                   labels={"reading_date":"Date","water_quantity":"Water (m³)"},
                   title=f"{jm} — Daily Water Supplied (Last 7 Days)")
        st.plotly_chart(fig,use_container_width=True)
        st.markdown(f"**Total:** {daily['water_quantity'].sum():.2f} m³  **Days Updated:** {(daily['water_quantity']>0).sum()}/7")
    if st.button("Close View"): st.session_state["selected_jalmitra"]=None

# ---------------------------  NEW — Daily BFM Readings  ---------------------------
st.markdown("---"); st.subheader("📅 BFM Readings Updated Today")
if today_upd.empty:
    st.info("No BFM readings recorded today.")
else:
    # ensure all required columns exist
    for c in ["jalmitra","scheme_name","reading","water_quantity"]:
        if c not in today_upd.columns: today_upd[c]=""
    daily=today_upd[["jalmitra","scheme_name","reading","water_quantity"]].copy()
    daily.columns=["Jalmitra","Scheme Name","BFM Reading","Water Quantity (m³)"]
    daily=daily.sort_values("Jalmitra")
    st.dataframe(
        daily.style.format({"BFM Reading":"{:06d}","Water Quantity (m³)":"{:.2f}"})
        .background_gradient(cmap="Blues",subset=["BFM Reading","Water Quantity (m³)"]),
        height=350)

# ---------------------------  Export  ---------------------------
st.markdown("---"); st.subheader("📤 Export Snapshot")
st.download_button("Schemes CSV",st.session_state["schemes"].to_csv(index=False).encode("utf-8"),"schemes.csv")
st.download_button("Readings CSV",st.session_state["readings"].to_csv(index=False).encode("utf-8"),"readings.csv")
st.download_button("Metrics CSV",metrics.to_csv(index=False).encode("utf-8"),"metrics.csv")
st.success(f"Dashboard ready for SO {so}. Demo data generated: {st.session_state['demo_generated']}")
