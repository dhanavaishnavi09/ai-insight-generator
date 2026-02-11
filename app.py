"""import streamlit as st
import pandas as pd
from insights import generate_insights


st.set_page_config(page_title="AI Insight Mini", layout="centered")

st.title("AI Insight Generator (Mini)")
st.write("Upload a CSV and get simple, decision-style insights.")

st.caption("Expected columns: date, region, sales")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

use_sample = st.checkbox("Use sample data instead")


df = None

if use_sample:
    #df = pd.read_csv("data/sample_sales.csv")
    df = pd.read_csv("data/sample_sales.csv", encoding="utf-8-sig")

elif uploaded is not None:
    df = pd.read_csv(uploaded)

if df is not None:
    st.subheader("Preview")
    st.dataframe(df.head(10), use_container_width=True)

    if st.button("Generate Insights"):
        with st.spinner("Analyzing..."):
            insights = generate_insights(df)

        st.subheader("Insights")
        for i, line in enumerate(insights, start=1):
            st.write(f"{i}. {line}")
else:
    st.info("Upload a CSV or tick 'Use sample data instead'.")
    if st.button("Generate Insights"):
        try:
            with st.spinner("Analyzing..."):
                insights = generate_insights(df)

            st.subheader("Insights")
            for i, line in enumerate(insights, start=1):
                st.write(f"{i}. {line}")

        except Exception as e:
            st.error(str(e))
            st.info(f"Detected columns: {list(df.columns)}")"""

import streamlit as st
import pandas as pd
from insights import generate_insights


st.set_page_config(page_title="AI Insight Mini", layout="centered")

st.title("AI Insight Generator (Mini)")
st.write("Upload any CSV — the app will try to auto-detect date + numeric value + region columns.")
st.caption("Tip: works best if your CSV has a date/month column and a sales/revenue/amount column.")

uploaded = st.file_uploader("Upload CSV", type=["csv"])

use_sample = st.checkbox("Use sample data instead")

df = None
if use_sample:
    df = pd.read_csv("data/sample_sales.csv")
elif uploaded is not None:
    df = pd.read_csv(uploaded)

if df is None:
    st.info("Upload a CSV or tick 'Use sample data instead'.")
else:
    st.subheader("Preview")
    st.dataframe(df.head(15), use_container_width=True)
    st.write("Detected columns:", list(df.columns))

    if st.button("Generate Insights"):
        try:
            with st.spinner("Analyzing..."):
                insights = generate_insights(df)

            st.subheader("Insights")
            for i, line in enumerate(insights, start=1):
                st.write(f"{i}. {line}")

        except Exception as e:
            st.error(str(e))
            st.info("Fix suggestion: ensure your CSV has a date-like column and a numeric column.")
