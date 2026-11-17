"""
model_engine.py
Machine Learning and Predictive Analytics engine for TelcoX.
Implements the notebook's Random Forest model and provides real-time scoring.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

# Feature columns expected by baseline model
BASELINE_CATEGORICAL = ["segment", "plan_type"]
BASELINE_NUMERICAL = ["monthly_spend", "tenure_months", "nps_score"]

def train_baseline_model(customers_df):
    """
    Trains the exact Random Forest model from the notebook:
    Features: segment, monthly_spend, tenure_months, plan_type, nps_score
    Target: churn_6mo_flag
    """
    X_raw = customers_df[["segment", "monthly_spend", "tenure_months", "plan_type", "nps_score"]].copy()
    y = customers_df["churn_6mo_flag"].copy()

    # One-hot encode exactly as notebook
    X_encoded = pd.get_dummies(X_raw, columns=["segment", "plan_type"])
    
    # Ensure expected columns are present
    expected_cols = [
        "monthly_spend", "tenure_months", "nps_score",
        "segment_consumer", "segment_smallbiz",
        "plan_type_enterprise", "plan_type_prepaid", "plan_type_unlimited"
    ]
    for col in expected_cols:
        if col not in X_encoded.columns:
            X_encoded[col] = 0
    X_encoded = X_encoded[expected_cols]

    # Train / Test split matching notebook
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred, output_dict=True)

    # Feature importances
    feature_importance_df = pd.DataFrame({
        "Feature": expected_cols,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    # Clean feature names for corporate display
    name_map = {
        "monthly_spend": "Monthly Spend ($)",
        "tenure_months": "Customer Tenure (Months)",
        "nps_score": "Net Promoter Score (NPS)",
        "segment_consumer": "Segment: Consumer",
        "segment_smallbiz": "Segment: Small Business",
        "plan_type_enterprise": "Plan: Enterprise",
        "plan_type_unlimited": "Plan: Unlimited",
        "plan_type_prepaid": "Plan: Prepaid"
    }
    feature_importance_df["Feature_Name"] = feature_importance_df["Feature"].map(name_map).fillna(feature_importance_df["Feature"])

    # Score all customers in dataset
    all_probs = model.predict_proba(X_encoded)[:, 1]
    all_preds = model.predict(X_encoded)

    results_all = customers_df.copy()
    results_all["churn_probability"] = (all_probs * 100).round(1)
    results_all["predicted_churn"] = all_preds
    results_all["risk_tier"] = pd.cut(
        results_all["churn_probability"],
        bins=[-0.1, 35, 60, 100],
        labels=["Low Risk", "Moderate Risk", "High Risk"]
    )

    return {
        "model": model,
        "feature_cols": expected_cols,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_prob": y_prob,
        "accuracy": acc,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
        "classification_report": cr,
        "feature_importance": feature_importance_df,
        "scored_customers": results_all
    }

def predict_single_customer(model, feature_cols, segment, plan_type, monthly_spend, tenure_months, nps_score):
    """
    Predict churn probability for an individual customer using the trained model.
    """
    row = {
        "monthly_spend": float(monthly_spend),
        "tenure_months": int(tenure_months),
        "nps_score": int(nps_score),
        "segment_consumer": 1 if segment == "consumer" else 0,
        "segment_smallbiz": 1 if segment == "smallbiz" else 0,
        "plan_type_enterprise": 1 if plan_type == "enterprise" else 0,
        "plan_type_prepaid": 1 if plan_type == "prepaid" else 0,
        "plan_type_unlimited": 1 if plan_type == "unlimited" else 0,
    }
    df_in = pd.DataFrame([row])[feature_cols]
    prob = model.predict_proba(df_in)[0, 1]
    pred = int(prob >= 0.5)

    if prob < 0.35:
        tier = "Low Risk"
        badge_color = "#10B981" # Green
    elif prob < 0.60:
        tier = "Moderate Risk"
        badge_color = "#F59E0B" # Amber
    else:
        tier = "High Risk"
        badge_color = "#EF4444" # Red

    # Prescriptive logic
    if pred == 1 or prob >= 0.45:
        if nps_score <= 4 and monthly_spend > 100:
            recommendation = "High-Value Detractor: Escalate to VIP Retention Concierge, conduct immediate billing audit, and provide personalized loyalty credit."
        elif tenure_months < 12:
            recommendation = "Early-Tenure Onboarding Risk: Proactively deploy customer success onboarding check-in and 15% promotional plan credit."
        elif plan_type == "prepaid":
            recommendation = "Prepaid Churn Risk: Offer subsidized migration to unlimited postpaid plan with device discount bonus."
        else:
            recommendation = "Retention Benefit Package: Offer retention benefits, tailored plan optimization, and loyalty reward points."
    else:
        recommendation = "Healthy Account: Maintain regular engagement cadences, automated satisfaction touchpoints, and explore cross-sell opportunities."

    return {
        "probability": round(prob * 100, 1),
        "prediction": pred,
        "tier": tier,
        "badge_color": badge_color,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    from data_processor import clean_data
    customers, _, _, _ = clean_data()
    results = train_baseline_model(customers)
    print("Baseline Model Accuracy:", results["accuracy"])
    print("Baseline ROC-AUC:", results["roc_auc"])
    print("Feature Importance:")
    print(results["feature_importance"])
    sample = predict_single_customer(results["model"], results["feature_cols"], "consumer", "unlimited", 120.0, 12, 2)
    print("Sample prediction:", sample)
