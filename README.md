# TelcoX Customer Analytics Dashboard
### Analytics Maturity Assessment (DIAI Framework + Cognitive Analytics)

A corporate-grade, interactive analytics dashboard built with **Python**, **Streamlit**, and **Plotly** to assess customer retention, support center friction, retail store interactions, and network quality for **TelcoX**.

---

## 📌 Problem Statement & Analytics Maturity Model
TelcoX faces a **31.0% customer churn rate** representing over **$5,800/month in recurring revenue at risk** across its subscriber base. This project evaluates TelcoX across the 5 tiers of the **Analytics Maturity Spectrum**:

1. **Level 1: Descriptive Analytics ("What happened?")**
   - Churn rate, monthly spend, tenure, and customer satisfaction (NPS) cohorts across consumer, enterprise, and prepaid plans.
2. **Level 2: Diagnostic Analytics ("Why did it happen?")**
   - Root-cause deep dive correlating call center wait times, unresolved service tickets, retail store visit types, and cell tower packet loss/latency with churn.
3. **Level 3: Predictive Analytics ("What will happen?")**
   - Implementation and explainability of the Random Forest machine learning classifier (reproducing notebook results: 62.5% accuracy, feature importance ranking), coupled with a **Live Interactive Churn Risk Simulator**.
4. **Level 4: Prescriptive Analytics ("What should we do?")**
   - Actionable retention playbooks (VIP Concierge, Support Fatigue, and Prepaid Migration) linked to root causes, plus a dynamic **Campaign ROI and Revenue Preservation Calculator**.
5. **Level 5: Cognitive Analytics ("The Missing Tier")**
   - Comprehensive architectural blueprint of autonomous, self-learning AI systems (real-time voice emotion detection, autonomous self-healing networks, and reinforcement learning Next-Best-Action engines) required to achieve zero-latency customer retention.

---

## 🚀 Quick Start Guide

### 1. Launch the Dashboard
Run the provided shell script or start Streamlit directly:
```bash
./run_dashboard.sh
```
Or with the virtual environment binary:
```bash
./venv/bin/streamlit run app.py
```

Open your browser to: **`http://localhost:8501`**

---

## 📁 Project Architecture & Files
* `app.py`: Main interactive Streamlit application with custom corporate styling, sidebar filters, 5 analytics maturity modules, and a unified Customer 360 viewer.
* `data_processor.py`: Data ingestion and cleaning module. Safely reads from the original Excel file without modifying or deleting any source data.
* `model_engine.py`: Machine learning engine that trains the baseline Random Forest model, computes performance metrics, feature importances, and live risk inferences.
* `Case Study 1- DIAI framework - TelcoX Case Study Data.xlsx`: Original dataset containing `Customers`, `Calls`, `NetworkEvents`, and `Store_Visits` sheets.
* `run_dashboard.sh`: Convenience launcher script.
* `venv/`: Local Python 3.9 virtual environment containing Streamlit, Plotly, Scikit-learn, Pandas, and Openpyxl.

---

## 📊 Core Datasets
* **Customers (200 records)**: Demographics, monthly spend, tenure, plan type, NPS, and 6-month churn flag.
* **Calls (5,000 records)**: Touchpoint logs detailing call reason, agent ID, hold time, duration, and resolution status.
* **NetworkEvents (1,000 records)**: Cell tower incidents capturing packet loss and transmission latency.
* **Store_Visits (300 records)**: Physical store visits tracking issue types, duration, and in-store purchase value.
