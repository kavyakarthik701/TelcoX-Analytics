"""
app.py
TelcoX Customer Analytics Dashboard
Analytics Maturity Assessment (DIAI Framework + Cognitive Analytics)

Author: Kavya Karthik
College Project: TelcoX Customer Retention & Analytics Maturity Assessment
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import precision_score, recall_score

from data_processor import clean_data, get_customer_360_features
from model_engine import train_baseline_model, predict_single_customer

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="TelcoX | Customer Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Global CSS
# ---------------------------------------------------------
st.markdown("""
<style>
    .stApp {
        background-color: #F5F3EE;
        color: #1A1A2E;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F172A 0%, #1E293B 100%) !important;
        box-shadow: 2px 0 8px rgba(0,0,0,0.25);
    }
    section[data-testid="stSidebar"] * { color: #E2E8F0 !important; }
    section[data-testid="stSidebar"] .stRadio label {
        padding: 9px 14px !important;
        border-radius: 6px !important;
        color: #94A3B8 !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        display: block;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background-color: rgba(255,255,255,0.08) !important;
        color: #60A5FA !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.1) !important;
        margin: 12px 0 !important;
    }

    .welcome-banner {
        background: linear-gradient(135deg, #1E3A5F 0%, #164E63 50%, #134E4A 100%);
        border-radius: 10px;
        padding: 24px 30px;
        margin-bottom: 22px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.12);
    }
    .welcome-banner h1 { font-size: 24px; font-weight: 800; color: #F8FAFC; margin: 0 0 5px 0; }
    .welcome-banner p  { font-size: 13px; color: #94A3B8; margin: 0; line-height: 1.5; }
    .welcome-badge {
        display: inline-block;
        background: rgba(255,255,255,0.12);
        border: 1px solid rgba(255,255,255,0.2);
        border-radius: 20px;
        padding: 3px 12px;
        font-size: 11px; font-weight: 600; color: #7DD3FC;
        margin-bottom: 10px; letter-spacing: 0.5px; text-transform: uppercase;
    }

    .stats-strip {
        display: flex; justify-content: space-between; align-items: center;
        background: #FFFFFF; border: 1px solid #E8E5DF; border-radius: 8px;
        padding: 16px 20px; margin-bottom: 20px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .stat-cell { text-align: center; flex: 1; border-right: 1px solid #EDE9E3; padding: 2px 10px; }
    .stat-cell:last-child { border-right: none; }
    .stat-number { font-size: 22px; font-weight: 800; color: #1A1A2E; line-height: 1.1; display: block; }
    .stat-label  { font-size: 10.5px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.6px; color: #78716C; margin-top: 4px; display: block; }

    .section-title    { font-size: 21px; font-weight: 800; color: #0F172A; margin: 0 0 2px 0; }
    .section-subtitle { font-size: 13px; color: #64748B; margin: 0 0 10px 0; }

    .accent-line { border: 0; height: 3px; border-radius: 2px; margin: 8px 0 20px 0; }
    .accent-blue     { background: linear-gradient(90deg, #2563EB, #60A5FA); }
    .accent-teal     { background: linear-gradient(90deg, #0D9488, #34D399); }
    .accent-lavender { background: linear-gradient(90deg, #7C3AED, #A78BFA); }
    .accent-peach    { background: linear-gradient(90deg, #EA580C, #FB923C); }
    .accent-coral    { background: linear-gradient(90deg, #DC2626, #F87171); }

    .panel { border-radius: 7px; padding: 14px 18px; margin-bottom: 16px; }
    .panel-teal     { background: #ECFDF5; border-left: 4px solid #0D9488; }
    .panel-blue     { background: #EFF6FF; border-left: 4px solid #2563EB; }
    .panel-lavender { background: #F5F3FF; border-left: 4px solid #7C3AED; }
    .panel-peach    { background: #FFF7ED; border-left: 4px solid #EA580C; }
    .panel-coral    { background: #FFF1F2; border-left: 4px solid #DC2626; }
    .panel-amber    { background: #FFFBEB; border-left: 4px solid #D97706; }

    .process-box {
        background: #FFFFFF; border: 1px solid #E8E5DF; border-radius: 8px;
        padding: 16px 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .process-badge {
        display: inline-block; padding: 3px 10px; border-radius: 4px;
        font-size: 10px; font-weight: 700; color: #FFFFFF;
        margin-bottom: 8px; letter-spacing: 0.5px; text-transform: uppercase;
    }

    div[data-testid="stSidebarNav"] { display: none; }
    .stDeployButton { display: none; }
    footer { display: none; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Data & Model
# ---------------------------------------------------------
@st.cache_data
def load_data():
    customers, calls, network, visits = clean_data()
    df_360 = get_customer_360_features(customers, calls, visits)
    return customers, calls, network, visits, df_360

@st.cache_resource
def load_ml_model(customers_df):
    return train_baseline_model(customers_df)

customers, calls, network, visits, df_360 = load_data()
model_bundle = load_ml_model(customers)

y_pred = model_bundle["model"].predict(model_bundle["X_test"])
y_test  = model_bundle["y_test"]
MODEL_PRECISION = precision_score(y_test, y_pred, zero_division=0)
MODEL_RECALL    = recall_score(y_test, y_pred, zero_division=0)

CLR_BLUE     = "#2563EB"
CLR_TEAL     = "#0D9488"
CLR_LAVENDER = "#7C3AED"
CLR_PEACH    = "#EA580C"
CLR_CORAL    = "#DC2626"
CLR_GREEN    = "#059669"
CHURN_COLORS = {"Retained": CLR_BLUE, "Churned": CLR_CORAL}

# ---------------------------------------------------------
# Chart Helper — clean, no clipping
# ---------------------------------------------------------
def style_plot(fig, title="", height=320, margin_t=50, margin_b=40):
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=13, color="#0F172A", family="sans-serif", weight="bold"),
            x=0.0, y=0.98, pad=dict(l=2)
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="sans-serif", size=11, color="#374151"),
        margin=dict(l=16, r=20, t=margin_t, b=margin_b),
        height=height,
        hoverlabel=dict(bgcolor="#0F172A", font_size=11, font_color="#F1F5F9"),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(size=11), bgcolor="rgba(0,0,0,0)"
        )
    )
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor="#EDE9E3", zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor="#EDE9E3", zeroline=False, tickfont=dict(size=11))
    return fig

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:16px 4px 18px 4px;">
        <div style="font-size:11px;letter-spacing:2px;text-transform:uppercase;color:#475569;margin-bottom:6px;">College Project</div>
        <div style="font-size:19px;font-weight:800;color:#F8FAFC;line-height:1.2;">TelcoX Analytics</div>
        <div style="font-size:11px;color:#64748B;margin-top:3px;">Analytics Maturity Assessment</div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")

    selected_section = st.radio(
        "Navigation",
        ["Overview", "Descriptive Analytics", "Diagnostic Analytics",
         "Predictive Analytics", "Prescriptive Analytics", "Cognitive Analytics"],
        index=0, label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("""
    <div style="font-size:11px;color:#64748B;line-height:1.9;padding:0 2px;">
        <div style="color:#94A3B8;font-weight:600;margin-bottom:4px;">DATASET SUMMARY</div>
        <span style="color:#CBD5E1;">Total Customers</span><span style="float:right;color:#E2E8F0;">200</span><br>
        <span style="color:#CBD5E1;">Churn Rate</span><span style="float:right;color:#F87171;">31.0%</span><br>
        <span style="color:#CBD5E1;">Model</span><span style="float:right;color:#E2E8F0;">Random Forest</span><br>
        <span style="color:#CBD5E1;">Accuracy</span><span style="float:right;color:#34D399;">62.5%</span>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# 1. OVERVIEW
# ===========================================================
if selected_section == "Overview":

    st.markdown("""
    <div class="welcome-banner">
        <div class="welcome-badge">Analytics Maturity Assessment · DIAI Framework</div>
        <h1>TelcoX | Customer Analytics</h1>
        <p>Systematic maturity classification of TelcoX analytical activities across Descriptive, Diagnostic,
        Predictive, Prescriptive, and Cognitive Analytics — applied to a 200-subscriber churn dataset.</p>
    </div>
    """, unsafe_allow_html=True)

    total_cust = len(customers)
    churn_cnt  = int(customers["churn_6mo_flag"].sum())
    retain_cnt = total_cust - churn_cnt
    churn_pct  = churn_cnt / total_cust * 100
    retain_pct = 100 - churn_pct
    avg_nps    = customers["nps_score"].mean()

    # 5-cell stats strip (not 7 — prevents squeeze)
    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-cell">
            <span class="stat-number">200</span>
            <span class="stat-label">Total Customers</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_CORAL};">{churn_cnt}</span>
            <span class="stat-label">Churned Customers</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_TEAL};">{retain_cnt}</span>
            <span class="stat-label">Retained Customers</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_CORAL};">{churn_pct:.1f}%</span>
            <span class="stat-label">Churn Rate</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_TEAL};">{retain_pct:.1f}%</span>
            <span class="stat-label">Retention Rate</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1.4])

    with col1:
        # Donut — legend only, no outside labels (avoids overlap)
        churn_df = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Count":  [retain_cnt, churn_cnt]
        })
        fig_pie = px.pie(
            churn_df, names="Status", values="Count",
            color="Status",
            color_discrete_map={"Retained": CLR_BLUE, "Churned": CLR_CORAL},
            hole=0.55
        )
        fig_pie.update_traces(
            textinfo="percent",
            textposition="inside",
            textfont=dict(size=14, color="#FFFFFF", weight="bold"),
            marker=dict(line=dict(color="#F5F3EE", width=2.5)),
            pull=[0, 0.05]
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=10, r=10, t=46, b=10),
            height=300,
            showlegend=True,
            legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5, font=dict(size=12)),
            title=dict(text="Overall Churn Distribution", font=dict(size=13, color="#0F172A", weight="bold"))
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        # Maturity bar — text inside bars only, enough right margin
        stages_df = pd.DataFrame({
            "Level": ["Descriptive", "Diagnostic", "Predictive", "Prescriptive", "Cognitive\n(Proposed)"],
            "Score": [85, 70, 55, 35, 10],
            "Status": ["Implemented", "Implemented", "Implemented", "Partial", "Missing Gap"]
        })
        color_map = {"Implemented": CLR_TEAL, "Partial": CLR_PEACH, "Missing Gap": CLR_CORAL}
        fig_mat = px.bar(
            stages_df, x="Score", y="Level", orientation="h",
            color="Status", color_discrete_map=color_map,
            text=stages_df["Score"].apply(lambda v: f"{v}%"),
            category_orders={"Level": ["Cognitive\n(Proposed)", "Prescriptive", "Predictive", "Diagnostic", "Descriptive"]}
        )
        fig_mat.update_traces(
            textposition="inside",
            textfont=dict(size=12, color="#FFFFFF", weight="bold"),
            marker_line_width=0
        )
        fig_mat.update_layout(xaxis_range=[0, 100])
        style_plot(fig_mat, "Analytics Maturity Level Progression", height=300, margin_t=46, margin_b=10)
        st.plotly_chart(fig_mat, use_container_width=True)

    st.markdown(f"""
    <div class="panel panel-peach">
        <b style="color:#9A3412;">Project Problem Statement</b><br>
        <span style="font-size:12.5px;color:#374151;line-height:1.6;">
        TelcoX faces a <b>31.0% Customer Churn Rate</b> across its 200-subscriber cohort.
        This dashboard classifies the company's analytical activities into the DIAI maturity levels and
        identifies <b>Cognitive Analytics</b> as the missing level whose implementation would enable
        autonomous, real-time customer retention.
        </span>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# 2. DESCRIPTIVE ANALYTICS
# ===========================================================
elif selected_section == "Descriptive Analytics":

    st.markdown('<div class="section-title">Descriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">What happened? — Customer churn patterns across plan types and satisfaction scores.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="accent-line accent-blue">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # Churn by Plan Type — no outside text, use hover
        plan_df = customers.groupby(["plan_type", "churn_6mo_flag"]).size().reset_index(name="Subscribers")
        plan_df["Churn Status"] = plan_df["churn_6mo_flag"].map({0: "Retained", 1: "Churned"})
        plan_df["Plan Type"] = plan_df["plan_type"].str.title()

        fig_plan = px.bar(
            plan_df, x="Plan Type", y="Subscribers",
            color="Churn Status", barmode="group",
            color_discrete_map=CHURN_COLORS,
            labels={"Subscribers": "Number of Subscribers"},
            text="Subscribers"
        )
        # auto text — Plotly places inside or outside based on space
        fig_plan.update_traces(textposition="auto", textfont_size=12, cliponaxis=False)
        style_plot(fig_plan, "Churn by Plan Type", height=320, margin_t=50, margin_b=40)
        st.plotly_chart(fig_plan, use_container_width=True)

    with col2:
        # Churn by Segment
        seg_df = customers.copy()
        seg_df["Segment"] = seg_df["segment"].map({"consumer": "Consumer", "smallbiz": "Small Business"})
        seg_grp = seg_df.groupby(["Segment", "churn_6mo_flag"]).size().reset_index(name="Subscribers")
        seg_grp["Churn Status"] = seg_grp["churn_6mo_flag"].map({0: "Retained", 1: "Churned"})

        fig_seg = px.bar(
            seg_grp, x="Segment", y="Subscribers",
            color="Churn Status", barmode="group",
            color_discrete_map=CHURN_COLORS,
            labels={"Subscribers": "Number of Subscribers"},
            text="Subscribers"
        )
        fig_seg.update_traces(textposition="auto", textfont_size=12, cliponaxis=False)
        style_plot(fig_seg, "Churn by Customer Segment", height=320, margin_t=50, margin_b=40)
        st.plotly_chart(fig_seg, use_container_width=True)

    # NPS distribution full-width — the most informative satisfaction chart
    nps_df = customers.copy()
    nps_df["Churn Status"] = nps_df["churn_6mo_flag"].map({0: "Retained", 1: "Churned"})
    fig_nps = px.histogram(
        nps_df, x="nps_score", color="Churn Status",
        barmode="group", nbins=11,
        color_discrete_map=CHURN_COLORS,
        labels={"nps_score": "Net Promoter Score (0–10)", "count": "Number of Customers"}
    )
    fig_nps.update_layout(bargap=0.15)
    style_plot(fig_nps, "Customer Satisfaction (NPS) Distribution — Retained vs Churned", height=300, margin_t=46, margin_b=40)
    st.plotly_chart(fig_nps, use_container_width=True)

    st.markdown("""
    <div class="panel panel-blue">
        <b style="color:#1D4ED8;">Key Descriptive Findings:</b>
        <ul style="font-size:12.5px;color:#1E293B;margin:6px 0 0 0;padding-left:18px;line-height:1.9;">
            <li>Prepaid subscribers show the highest churn propensity (~40% churn rate).</li>
            <li>Consumer and Small Business segments share a comparable churn rate of approximately 31%.</li>
            <li>Churned customers have lower average NPS scores (3.87) vs retained customers (4.33).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# 3. DIAGNOSTIC ANALYTICS
# ===========================================================
elif selected_section == "Diagnostic Analytics":

    st.markdown('<div class="section-title">Diagnostic Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Why did it happen? — Root causes of customer dissatisfaction and service friction.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="accent-line accent-teal">', unsafe_allow_html=True)

    retained_df = customers[customers["churn_6mo_flag"] == 0]
    churned_df  = customers[customers["churn_6mo_flag"] == 1]
    nps_ret  = retained_df["nps_score"].mean()
    nps_chu  = churned_df["nps_score"].mean()
    ten_ret  = retained_df["tenure_months"].mean()
    ten_chu  = churned_df["tenure_months"].mean()
    avg_hold = calls["hold_time_sec"].mean()
    unresolved_per_cust = calls[calls["resolved_flag"] == 0].groupby("customer_id").size().mean()

    # 4-cell strip — fits without squeezing
    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_TEAL};">4.33 / 10</span>
            <span class="stat-label">NPS — Retained</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_CORAL};">3.87 / 10</span>
            <span class="stat-label">NPS — Churned</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_BLUE};">{ten_ret:.1f} mo</span>
            <span class="stat-label">Avg Tenure — Retained</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_CORAL};">{ten_chu:.1f} mo</span>
            <span class="stat-label">Avg Tenure — Churned</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_PEACH};">{avg_hold:.0f}s</span>
            <span class="stat-label">Avg Hold Time</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_CORAL};">{unresolved_per_cust:.1f}</span>
            <span class="stat-label">Unresolved Calls/Customer</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        # NPS grouped bar — wide bars, text auto
        nps_comp = pd.DataFrame({
            "Status": ["Retained", "Churned"],
            "Avg NPS": [round(nps_ret, 2), round(nps_chu, 2)]
        })
        fig_nps = px.bar(
            nps_comp, x="Status", y="Avg NPS",
            color="Status",
            color_discrete_map={"Retained": CLR_BLUE, "Churned": CLR_CORAL},
            text="Avg NPS",
            labels={"Avg NPS": "Average NPS Score (0–10)"}
        )
        fig_nps.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside",
            textfont_size=14,
            width=0.35
        )
        fig_nps.update_layout(showlegend=False, yaxis_range=[0, 11])
        style_plot(fig_nps, "Average NPS: Retained vs Churned Customers", height=320, margin_t=46, margin_b=40)
        st.plotly_chart(fig_nps, use_container_width=True)

    with col2:
        # Support call reasons — stacked bar, no text labels on bars
        call_df = calls.groupby(["call_reason_tag", "resolved_flag"]).size().reset_index(name="Calls")
        call_df["Resolution"] = call_df["resolved_flag"].map({1: "Resolved", 0: "Unresolved"})
        call_df["Reason"] = call_df["call_reason_tag"].str.title()

        fig_calls = px.bar(
            call_df, x="Reason", y="Calls",
            color="Resolution", barmode="stack",
            color_discrete_map={"Resolved": CLR_TEAL, "Unresolved": CLR_CORAL},
            labels={"Reason": "Call Reason", "Calls": "Total Support Calls"}
        )
        style_plot(fig_calls, "Support Call Volume by Reason & Resolution Status", height=320, margin_t=46, margin_b=40)
        st.plotly_chart(fig_calls, use_container_width=True)

    st.markdown(f"""
    <div class="panel panel-teal">
        <b style="color:#065F46;">Root Cause Summary:</b>
        <ul style="font-size:12.5px;color:#1E293B;margin:6px 0 0 0;padding-left:18px;line-height:1.9;">
            <li>Churned customers have a lower average NPS (3.87 vs 4.33) — indicating pre-existing dissatisfaction.</li>
            <li>Shorter average tenure in churned customers (58.3 mo vs 65.8 mo) confirms early-stage risk.</li>
            <li>Average support queue hold time of <b>{avg_hold:.0f} seconds (~{avg_hold/60:.1f} min)</b> adds friction.</li>
            <li>Billing and Network enquiries generate the highest volume of unresolved support calls.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ===========================================================
# 4. PREDICTIVE ANALYTICS
# ===========================================================
elif selected_section == "Predictive Analytics":

    st.markdown('<div class="section-title">Predictive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">What is likely to happen? — Random Forest churn prediction model performance and feature analysis.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="accent-line accent-lavender">', unsafe_allow_html=True)

    # 4-cell model metrics strip
    st.markdown(f"""
    <div class="stats-strip">
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_BLUE};">62.5%</span>
            <span class="stat-label">Model Accuracy</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_TEAL};">{MODEL_PRECISION*100:.1f}%</span>
            <span class="stat-label">Precision (Churn)</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_PEACH};">{MODEL_RECALL*100:.1f}%</span>
            <span class="stat-label">Recall (Churn)</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number" style="color:{CLR_LAVENDER};">{model_bundle['roc_auc']:.3f}</span>
            <span class="stat-label">ROC-AUC Score</span>
        </div>
        <div class="stat-cell">
            <span class="stat-number">160 / 40</span>
            <span class="stat-label">Train / Test Split</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1])

    with col1:
        # Feature importance — text inside bars, right margin for longest label
        feat_df = model_bundle["feature_importance"].copy().sort_values("Importance", ascending=True)
        feat_df["Feature Display"] = feat_df["Feature_Name"].str.replace("($)", "(USD)", regex=False)

        fig_feat = px.bar(
            feat_df, x="Importance", y="Feature Display", orientation="h",
            color="Importance",
            color_continuous_scale=["#60A5FA", CLR_LAVENDER],
            labels={"Importance": "Gini Importance", "Feature Display": ""}
        )
        # Put value as text inside each bar — no clipping
        fig_feat.update_traces(
            text=feat_df["Importance"].apply(lambda v: f"{v:.3f}"),
            textposition="inside",
            textfont=dict(size=11, color="#FFFFFF", weight="bold"),
            cliponaxis=False
        )
        fig_feat.update_layout(coloraxis_showscale=False, showlegend=False)
        # Give enough left margin for feature labels
        style_plot(fig_feat, "Feature Importance Ranking (Random Forest)", height=330, margin_t=46, margin_b=10)
        fig_feat.update_layout(margin=dict(l=170, r=30, t=46, b=10))
        st.plotly_chart(fig_feat, use_container_width=True)

    with col2:
        # Confusion matrix — simple, clean
        cm = model_bundle["confusion_matrix"]
        fig_cm = px.imshow(
            cm, text_auto=True,
            x=["Pred: Retained", "Pred: Churned"],
            y=["Actual: Retained", "Actual: Churned"],
            color_continuous_scale=["#EFF6FF", "#2563EB"],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            aspect="auto"
        )
        fig_cm.update_traces(
            texttemplate="<b>%{z}</b>",
            textfont=dict(size=22, color="#0F172A")
        )
        fig_cm.update_layout(coloraxis_showscale=False)
        style_plot(fig_cm, "Confusion Matrix — Test Set (40 Customers)", height=330, margin_t=46, margin_b=50)
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("""
    <div class="panel panel-amber">
        <b style="color:#92400E;">Model Limitation Note:</b>
        <span style="font-size:12.5px;color:#374151;">
        The Random Forest model achieves <b>62.5% overall accuracy</b> on the 40-customer holdout set.
        Low <b>Recall of 16.7%</b> means only 2 of 12 actual churners are correctly identified.
        Monthly Charges, Customer Tenure, and NPS Score are the top three predictive features.
        A larger training dataset would significantly improve churn recall.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Interactive Scoring Tool
    st.markdown('<div class="section-title" style="font-size:17px;margin-top:8px;">Interactive Churn Probability Scoring</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Adjust customer attributes to generate a predicted churn probability (model prediction — not actual data).</div>', unsafe_allow_html=True)

    s1, s2 = st.columns([1.3, 1])
    with s1:
        c1, c2 = st.columns(2)
        with c1:
            sim_seg  = st.selectbox("Customer Segment", ["consumer", "smallbiz"],
                                     format_func=lambda x: "Consumer" if x == "consumer" else "Small Business")
            sim_plan = st.selectbox("Plan Type", ["unlimited", "prepaid", "enterprise"],
                                     format_func=str.title)
        with c2:
            sim_spend = st.slider("Monthly Charges (USD)", 20.0, 300.0, 115.0, 5.0)
            sim_nps   = st.slider("Net Promoter Score (0–10)", 0, 10, 4)
        sim_tenure = st.slider("Customer Tenure (Months)", 1, 72, 28)

    with s2:
        sim_res = predict_single_customer(
            model_bundle["model"], model_bundle["feature_cols"],
            sim_seg, sim_plan, sim_spend, sim_tenure, sim_nps
        )
        st.markdown(f"""
        <div style="background:#FFFFFF;border:1px solid #E8E5DF;border-radius:8px;padding:20px;box-shadow:0 1px 4px rgba(0,0,0,0.05);">
            <div style="font-size:10px;font-weight:700;color:#94A3B8;text-transform:uppercase;letter-spacing:1px;">Predicted Churn Risk</div>
            <div style="font-size:40px;font-weight:900;color:{sim_res['badge_color']};margin:6px 0 2px 0;">{sim_res['probability']}%</div>
            <div style="font-size:14px;font-weight:600;color:#374151;margin-bottom:12px;">Risk Level: {sim_res['tier']}</div>
            <div style="font-size:11px;color:#64748B;border-top:1px solid #EDE9E3;padding-top:10px;">
                <b style="color:#0F172A;">Recommended Action:</b><br>
                <span style="line-height:1.6;">{sim_res['recommendation']}</span>
            </div>
            <div style="font-size:10px;color:#94A3B8;margin-top:8px;font-style:italic;">* Predicted risk — not actual customer data.</div>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================
# 5. PRESCRIPTIVE ANALYTICS
# ===========================================================
elif selected_section == "Prescriptive Analytics":

    st.markdown('<div class="section-title">Prescriptive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">What should be done? — Targeted retention strategies based on predictive churn signals.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="accent-line accent-peach">', unsafe_allow_html=True)

    pb1, pb2, pb3 = st.columns(3)

    with pb1:
        st.markdown("""
        <div class="panel panel-coral">
            <b style="color:#991B1B;font-size:13.5px;">Strategy 1 — VIP Detractor Recovery</b><br>
            <span style="font-size:11px;color:#64748B;display:block;margin:4px 0 8px 0;">
                <b>Trigger:</b> Monthly Charges &gt; $100 &amp; NPS ≤ 4
            </span>
            <ul style="font-size:12px;color:#1E293B;line-height:1.9;margin:0;padding-left:16px;">
                <li>Assign a dedicated account manager.</li>
                <li>Proactive billing dispute audit within 48 hours.</li>
                <li>$25 loyalty credit applied to the next billing cycle.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with pb2:
        st.markdown("""
        <div class="panel panel-peach">
            <b style="color:#9A3412;font-size:13.5px;">Strategy 2 — Support Fatigue Relief</b><br>
            <span style="font-size:11px;color:#64748B;display:block;margin:4px 0 8px 0;">
                <b>Trigger:</b> Unresolved Calls ≥ 5 or Hold Time &gt; 300s
            </span>
            <ul style="font-size:12px;color:#1E293B;line-height:1.9;margin:0;padding-left:16px;">
                <li>Automatic priority queue routing on next contact.</li>
                <li>Supervisor callback guaranteed within 4 hours.</li>
                <li>60-day connection fee waiver applied.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with pb3:
        st.markdown("""
        <div class="panel panel-blue">
            <b style="color:#1E40AF;font-size:13.5px;">Strategy 3 — Prepaid Migration Offer</b><br>
            <span style="font-size:11px;color:#64748B;display:block;margin:4px 0 8px 0;">
                <b>Trigger:</b> Prepaid Plan &amp; Tenure &lt; 24 Months
            </span>
            <ul style="font-size:12px;color:#1E293B;line-height:1.9;margin:0;padding-left:16px;">
                <li>15% discount for a 12-month postpaid contract.</li>
                <li>Free SIM upgrade with 10 GB bonus data.</li>
                <li>Zero activation fees for the new plan.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROI Simulator
    st.markdown('<div class="section-title" style="font-size:17px;">Retention Campaign ROI Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">Estimate financial impact based on campaign reach and success rate.</div>', unsafe_allow_html=True)

    r1, r2 = st.columns([1, 1.2])
    target_churners = int(customers["churn_6mo_flag"].sum())
    risk_mrr = customers[customers["churn_6mo_flag"] == 1]["monthly_spend"].sum()

    with r1:
        camp_reach     = st.slider("Campaign Reach — % of Churned Customers Contacted", 20, 100, 75, 5)
        camp_success   = st.slider("Retention Success Rate (%)", 10, 60, 30, 5)
        incentive_cost = st.number_input("Incentive Cost per Customer (USD)", value=30.0, step=5.0)

    with r2:
        targeted_n      = int(target_churners * (camp_reach / 100))
        saved_n         = int(targeted_n * (camp_success / 100))
        avg_churn_spend = risk_mrr / target_churners if target_churners > 0 else 0
        saved_monthly   = saved_n * avg_churn_spend
        saved_annual    = saved_monthly * 12
        total_cost      = targeted_n * incentive_cost
        net_roi         = ((saved_annual - total_cost) / total_cost * 100) if total_cost > 0 else 0

        st.markdown(f"""
        <div class="stats-strip" style="margin-top:14px;">
            <div class="stat-cell">
                <span class="stat-number" style="color:{CLR_BLUE};">{targeted_n}</span>
                <span class="stat-label">Customers Targeted</span>
            </div>
            <div class="stat-cell">
                <span class="stat-number" style="color:{CLR_GREEN};">{saved_n}</span>
                <span class="stat-label">Accounts Retained</span>
            </div>
            <div class="stat-cell">
                <span class="stat-number" style="color:{CLR_TEAL};">${saved_monthly:,.0f}</span>
                <span class="stat-label">Monthly Revenue Saved</span>
            </div>
            <div class="stat-cell">
                <span class="stat-number" style="color:{CLR_GREEN};">${saved_annual:,.0f}</span>
                <span class="stat-label">Annual Revenue Preserved</span>
            </div>
            <div class="stat-cell">
                <span class="stat-number" style="color:{CLR_PEACH};">{net_roi:.0f}%</span>
                <span class="stat-label">Projected Net ROI</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ===========================================================
# 6. COGNITIVE ANALYTICS
# ===========================================================
elif selected_section == "Cognitive Analytics":

    st.markdown('<div class="section-title">Cognitive Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-subtitle">The missing level — Proposed autonomous feedback-and-learning architecture for real-time customer retention.</div>', unsafe_allow_html=True)
    st.markdown('<hr class="accent-line accent-lavender">', unsafe_allow_html=True)

    st.markdown("""
    <div class="panel panel-lavender">
        <b style="color:#5B21B6;font-size:13.5px;">Why Cognitive Analytics is Missing at TelcoX</b><br>
        <span style="font-size:12.5px;color:#1E293B;line-height:1.7;display:block;margin-top:6px;">
            TelcoX's current implementation covers Descriptive through Prescriptive levels —
            all of which are <b>retrospective and batch-oriented</b>: data is analysed after customer friction occurs
            and actions are triggered by static rules.
            <b>Cognitive Analytics</b> (Level 5) introduces an autonomous, self-learning AI layer that ingests
            real-time signals, reasons about customer intent, and acts within the same interaction to prevent churn.
        </span>
    </div>
    """, unsafe_allow_html=True)

    # 5-stage process boxes
    st.markdown('<div class="section-title" style="font-size:16px;margin:16px 0 10px 0;">Proposed Cognitive Feedback Process</div>', unsafe_allow_html=True)

    stages = [
        {"step": "Stage 1", "name": "Listen",      "color": CLR_BLUE,
         "desc": "Ingest real-time signals: live call audio, support tickets, network events, and in-app behaviour."},
        {"step": "Stage 2", "name": "Understand",  "color": CLR_TEAL,
         "desc": "Apply NLP to detect dissatisfaction, intent to cancel, and unmet expectations from each signal."},
        {"step": "Stage 3", "name": "Personalise", "color": CLR_LAVENDER,
         "desc": "Generate a Next-Best-Action recommendation tailored to each customer's risk profile and plan tier."},
        {"step": "Stage 4", "name": "Measure",     "color": CLR_PEACH,
         "desc": "Track customer response — retention outcome, NPS change, and revenue impact — in real time."},
        {"step": "Stage 5", "name": "Learn",       "color": CLR_CORAL,
         "desc": "Feed outcomes back into the model automatically, updating decision weights without manual retraining."},
    ]
    cols = st.columns(5)
    for col, s in zip(cols, stages):
        with col:
            st.markdown(f"""
            <div class="process-box" style="border-top:3px solid {s['color']};">
                <span class="process-badge" style="background:{s['color']};">{s['step']}</span>
                <div style="font-size:14px;font-weight:800;color:#0F172A;margin-bottom:6px;">{s['name']}</div>
                <p style="font-size:11px;color:#374151;line-height:1.55;margin:0;">{s['desc']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Maturity gap table
    st.markdown('<div class="section-title" style="font-size:16px;margin-bottom:8px;">Maturity Gap: TelcoX Today vs Proposed Cognitive Level</div>', unsafe_allow_html=True)

    gap_df = pd.DataFrame({
        "Capability": ["Data Telemetry", "Decision Latency", "Model Intelligence", "Action Execution", "Model Learning"],
        "TelcoX Today (Levels 1–4)": [
            "Batch relational tables (Excel / CSV)",
            "Days to weeks after friction occurs",
            "Random Forest — 62.5% accuracy, 16.7% recall",
            "Manual outbound calls with static rules",
            "Manual periodic retraining by analyst"
        ],
        "Proposed Cognitive Target (Level 5)": [
            "Real-time event streams (Kafka / IoT / Vector DB)",
            "Sub-second (<500 ms) during active interaction",
            "Multimodal LLMs + Contextual Reinforcement Learning",
            "Autonomous micro-incentive dispatch in-session",
            "Continuous closed-loop automated weight updates"
        ]
    })
    st.dataframe(gap_df, use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="panel panel-teal" style="margin-top:16px;">
        <b style="color:#065F46;">Impact of Implementing Cognitive Analytics:</b>
        <ul style="font-size:12.5px;color:#1E293B;margin:6px 0 0 0;padding-left:18px;line-height:1.9;">
            <li>Shift from retrospective batch analysis to real-time proactive intervention.</li>
            <li>Reduce churn by detecting dissatisfaction signals before a cancellation is filed.</li>
            <li>Improve NPS through personalised, timely outreach during service interactions.</li>
            <li>Enable the model to self-improve continuously — reducing the need for manual retraining.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
