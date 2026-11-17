"""
data_processor.py
Data loading, cleaning, and feature engineering for TelcoX Analytics Maturity Dashboard.
Does not modify any original source files.
"""

import pandas as pd
import numpy as np
import os

EXCEL_FILE = "Case Study 1- DIAI framework - TelcoX Case Study Data.xlsx"

def load_raw_data(filepath=EXCEL_FILE):
    """Load all 4 sheets from the Excel file."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    xls = pd.ExcelFile(filepath)
    raw_customers = pd.read_excel(xls, "Customers")
    raw_calls = pd.read_excel(xls, "Calls")
    raw_network = pd.read_excel(xls, "NetworkEvents")
    raw_visits = pd.read_excel(xls, "Store_Visits")
    return raw_customers, raw_calls, raw_network, raw_visits

def clean_data(filepath=EXCEL_FILE):
    """Clean the datasets by dropping Unnamed columns and auxiliary metadata columns."""
    raw_customers, raw_calls, raw_network, raw_visits = load_raw_data(filepath)

    def clean_columns(df, drop_col):
        # Drop columns starting with Unnamed
        df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
        df = df.drop(columns=[drop_col], errors="ignore")
        return df.copy()

    customers = clean_columns(raw_customers, "Who are our customers and how happy are they?")
    calls = clean_columns(raw_calls, "How do customers interact with the call center?")
    network = clean_columns(raw_network, "Did customers face network issues?")
    visits = clean_columns(raw_visits, "What happens when customers come to our stores?")

    # Clean data types
    customers["customer_id"] = customers["customer_id"].astype(int)
    customers["monthly_spend"] = customers["monthly_spend"].astype(float)
    customers["tenure_months"] = customers["tenure_months"].astype(int)
    customers["nps_score"] = customers["nps_score"].astype(int)
    customers["churn_6mo_flag"] = customers["churn_6mo_flag"].astype(int)

    # Parse timestamps
    calls["call_timestamp"] = pd.to_datetime(calls["call_timestamp"], format="mixed", errors="coerce")
    network["event_timestamp"] = pd.to_datetime(network["event_timestamp"], format="mixed", errors="coerce")
    visits["visit_timestamp"] = pd.to_datetime(visits["visit_timestamp"], format="mixed", errors="coerce")

    # Clean numerical columns
    calls["call_duration_sec"] = pd.to_numeric(calls["call_duration_sec"], errors="coerce")
    calls["hold_time_sec"] = pd.to_numeric(calls["hold_time_sec"], errors="coerce")
    calls["transfer_count"] = pd.to_numeric(calls["transfer_count"], errors="coerce")
    calls["resolved_flag"] = pd.to_numeric(calls["resolved_flag"], errors="coerce").fillna(0).astype(int)

    network["loss_percentage"] = pd.to_numeric(network["loss_percentage"], errors="coerce")
    network["avg_latency_ms"] = pd.to_numeric(network["avg_latency_ms"], errors="coerce")

    visits["visit_duration_min"] = pd.to_numeric(visits["visit_duration_min"], errors="coerce")
    visits["purchase_value"] = pd.to_numeric(visits["purchase_value"], errors="coerce").fillna(0.0)

    return customers, calls, network, visits

def get_customer_360_features(customers, calls, visits):
    """
    Aggregate calls and store visits to customer level and merge with customer profile.
    Provides diagnostic and enriched predictive features.
    """
    # Call center aggregations
    call_agg = calls.groupby("customer_id").agg(
        total_calls=("call_id", "count"),
        unresolved_calls=("resolved_flag", lambda x: (x == 0).sum()),
        resolved_calls=("resolved_flag", lambda x: (x == 1).sum()),
        avg_hold_time_sec=("hold_time_sec", "mean"),
        avg_call_duration_sec=("call_duration_sec", "mean"),
        avg_transfers=("transfer_count", "mean"),
        billing_calls=("call_reason_tag", lambda x: (x == "billing").sum()),
        network_calls=("call_reason_tag", lambda x: (x == "network").sum()),
        device_calls=("call_reason_tag", lambda x: (x == "device").sum()),
        upgrade_calls=("call_reason_tag", lambda x: (x == "upgrade").sum())
    ).reset_index()

    call_agg["resolution_rate"] = np.where(
        call_agg["total_calls"] > 0,
        (call_agg["resolved_calls"] / call_agg["total_calls"]) * 100,
        100.0
    )

    # Store visit aggregations
    visit_agg = visits.groupby("customer_id").agg(
        total_store_visits=("visit_id", "count"),
        total_store_spend=("purchase_value", "sum"),
        avg_visit_duration_min=("visit_duration_min", "mean")
    ).reset_index()

    # Merge with customers
    merged = customers.merge(call_agg, on="customer_id", how="left")
    merged = merged.merge(visit_agg, on="customer_id", how="left")

    # Fill NaN for customers with 0 store visits
    merged["total_store_visits"] = merged["total_store_visits"].fillna(0).astype(int)
    merged["total_store_spend"] = merged["total_store_spend"].fillna(0.0)
    merged["avg_visit_duration_min"] = merged["avg_visit_duration_min"].fillna(0.0)

    # Risk factor diagnostics
    merged["high_hold_time_flag"] = (merged["avg_hold_time_sec"] > 300).astype(int)
    merged["unresolved_issue_flag"] = (merged["unresolved_calls"] >= 5).astype(int)
    merged["detractor_flag"] = (merged["nps_score"] <= 6).astype(int)
    merged["promoter_flag"] = (merged["nps_score"] >= 9).astype(int)

    # Churn status label
    merged["churn_label"] = merged["churn_6mo_flag"].map({1: "Churned", 0: "Retained"})

    return merged

if __name__ == "__main__":
    customers, calls, network, visits = clean_data()
    df_360 = get_customer_360_features(customers, calls, visits)
    print("Cleaned datasets:")
    print(f"Customers: {customers.shape}")
    print(f"Calls: {calls.shape}")
    print(f"Network: {network.shape}")
    print(f"Visits: {visits.shape}")
    print(f"Customer 360 Dataset: {df_360.shape}")
