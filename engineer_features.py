# step_B_engineer_features.py
import pandas as pd
import numpy as np

df = pd.read_csv(r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\snapshot_features_churn_labels.csv", parse_dates=["last_purchase_date","first_purchase_date"])
# Add engineered features:
df["avg_order_value"] = df["avg_order_value"].fillna(0)
df["orders_per_tenure_month"] = df["orders_count"] / (df["tenure_days"]/30 + 1e-6)
df["spend_per_order"] = df["total_spent"] / (df["orders_count"].replace(0,1))
# bin monetary to log scale
df["log_monetary"] = np.log1p(df["total_spent"])
# flag high value
df["is_high_value"] = (df["total_spent"] > df["total_spent"].quantile(0.9)).astype(int)

# Save
df.to_csv(r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\features_for_model.csv", index=False)
print("Saved features_for_model.csv")
print(df.describe().T[['min','max','mean','std']])
