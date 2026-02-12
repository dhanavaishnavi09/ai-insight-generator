# 🚀 AI Insight Generator

> Turning raw sales data into structured, explainable business insights.

An interactive data intelligence tool built using Python and Streamlit that analyzes sales performance and automatically generates actionable insights for decision-makers.

---

## 🎯 Problem Statement

In most organizations, data exists in spreadsheets but insights require manual effort.

Business teams spend hours:
- Cleaning data
- Calculating KPIs
- Identifying trends
- Writing summaries

This project simulates how internal analytics tools in product companies transform raw data into decision-ready insights automatically.

---

## 🧠 What Makes This Different?

This is not just a dashboard.

It:
- Cleans messy column names
- Normalizes schema automatically
- Detects time-based trends
- Identifies best & worst performing regions
- Generates structured insight summaries

Built with **product-thinking mindset**, not just visualization.

---

## ⚙️ Architecture Overview
User → Streamlit UI → Data Cleaning Layer → Insight Engine → Business Summary Output

### Core Modules

- `app.py` → Frontend logic & UI
- `insights.py` → Data processing & insight engine
- `data/` → Sample dataset

---

## 📊 Key Features

✅ Automated schema normalization  
✅ Date parsing & time aggregation  
✅ Revenue KPI calculation  
✅ Region performance ranking  
✅ Insight narrative generation  
✅ Error handling for missing columns  

---

## 🛠️ Tech Stack

- Python 3.12
- Pandas
- Streamlit
- Git & GitHub

---

## 🚀 Live Demo

https://ai-insight-generator-uxtrwjbrvampnles336uwc.streamlit.app/

---

## ▶️ Run Locally

```bash
git clone https://github.com/dhanavaishnavi09/ai-insight-generator.git
cd ai-insight-generator
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m streamlit run app.py

