import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Marketing Prediction",
    page_icon="🏦",
    layout="centered",
)

# ── Load pipeline ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_pipeline():
    return joblib.load("best_model.pkl")   # ganti path sesuai lokasi file pipeline

pipeline = load_pipeline()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("🏦 Bank Marketing — Prediksi Langganan Deposito")
st.markdown("Isi data nasabah di bawah, lalu klik **Predict**.")

st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("📋 Data Numerik")
    age       = st.number_input("Age",       min_value=18,   max_value=95,   value=35)
    balance   = st.number_input("Balance",   min_value=-6847, max_value=81204, value=1000)
    day       = st.number_input("Day",       min_value=1,    max_value=31,   value=15)
    duration  = st.number_input("Duration (detik)", min_value=2, max_value=3881, value=200)
    campaign  = st.number_input("Campaign",  min_value=1,    max_value=63,   value=2)
    pdays     = st.number_input("Pdays",     min_value=-1,   max_value=854,  value=-1)
    previous  = st.number_input("Previous",  min_value=0,    max_value=58,   value=0)

with col2:
    st.subheader("🏷️ Data Kategorikal")
    job = st.selectbox("Job", [
        "admin.", "technician", "services", "management", "retired",
        "blue-collar", "unemployed", "entrepreneur", "housemaid",
        "unknown", "self-employed", "student"
    ])
    marital   = st.selectbox("Marital",   ["married", "single", "divorced"])
    education = st.selectbox("Education", ["secondary", "tertiary", "primary", "unknown"])
    default   = st.selectbox("Default",   ["no", "yes"])
    housing   = st.selectbox("Housing",   ["yes", "no"])
    loan      = st.selectbox("Loan",      ["no", "yes"])
    contact   = st.selectbox("Contact",   ["unknown", "cellular", "telephone"])
    month     = st.selectbox("Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    poutcome  = st.selectbox("Poutcome",  ["unknown", "other", "failure", "success"])

st.divider()

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("🔍 Predict", use_container_width=True, type="primary"):
    input_data = pd.DataFrame([{
        "age":       age,
        "job":       job,
        "marital":   marital,
        "education": education,
        "default":   default,
        "balance":   balance,
        "housing":   housing,
        "loan":      loan,
        "contact":   contact,
        "day":       day,
        "month":     month,
        "duration":  duration,
        "campaign":  campaign,
        "pdays":     pdays,
        "previous":  previous,
        "poutcome":  poutcome,
    }])

    pred       = pipeline.predict(input_data)[0]
    proba      = pipeline.predict_proba(input_data)[0]   # [prob_no, prob_yes]
    prob_yes   = proba[1]
    prob_no    = proba[0]

    st.divider()
    st.subheader("📊 Hasil Prediksi")

    # ── Verdict ──────────────────────────────────────────────────────────────
    if pred == "yes" or pred == 1:
        st.success("✅ Nasabah **AKAN** berlangganan deposito berjangka!")
    else:
        st.error("❌ Nasabah **TIDAK AKAN** berlangganan deposito berjangka.")

    # ── Probability cards ────────────────────────────────────────────────────
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="🟢 Probabilitas  YES",
            value=f"{prob_yes * 100:.2f} %",
            delta=None,
        )
    with c2:
        st.metric(
            label="🔴 Probabilitas  NO",
            value=f"{prob_no * 100:.2f} %",
            delta=None,
        )

    # ── Progress bar ─────────────────────────────────────────────────────────
    st.markdown("**Confidence bar (YES)**")
    st.progress(float(prob_yes))

    # ── Input recap ──────────────────────────────────────────────────────────
    with st.expander("📝 Lihat data yang diinput"):
        st.dataframe(input_data.T.rename(columns={0: "Value"}), use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Model: pipeline.pkl  |  Dataset: Bank Marketing UCI")