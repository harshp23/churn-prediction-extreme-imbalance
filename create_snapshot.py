
import pandas as pd
import numpy as np
from datetime import timedelta

# --- CONFIG ---
orders_fp = r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\clean_orders.csv"   #  cleaned orders file
order_items_fp = r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\olist_order_items_dataset.csv"
customers_fp = r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\olist_customers_dataset.csv"
snapshot_date = pd.to_datetime("2018-06-30")   # choose snapshot (must be < dataset max)
prediction_window_days = 180                   # label window (churn if no purchases in next 180 days)

# --- LOAD ---
orders = pd.read_csv(orders_fp, parse_dates=["order_purchase_timestamp", 
                                             "order_approved_at",
                                             "order_delivered_customer_date",
                                             "order_estimated_delivery_date"], low_memory=False)
order_items = pd.read_csv(order_items_fp)
customers = pd.read_csv(customers_fp)

# Use customer_unique_id as unique customer key
orders = orders.merge(customers[["customer_id","customer_unique_id"]], on="customer_id", how="left")

# Filter data up to snapshot_date for features
orders_hist = orders[orders["order_purchase_timestamp"] <= snapshot_date].copy()

# Future window for labeling
start_label = snapshot_date + pd.Timedelta(days=1)
end_label = snapshot_date + pd.Timedelta(days=prediction_window_days)

orders_future = orders[(orders["order_purchase_timestamp"] > snapshot_date) &
                       (orders["order_purchase_timestamp"] <= end_label)].copy()

# Merge order_items to compute monetary (join only on order ids in hist)
oi = order_items.copy()
oi['price'] = oi['price'].astype(float)
oi['freight_value'] = oi['freight_value'].astype(float)
oi['order_value'] = oi['price'] + oi['freight_value']

# Features aggregated up to snapshot_date (hist)
# total orders (frequency), total spend (monetary), avg order value, days since first purchase, etc.
agg_order_value = (orders_hist.merge(oi, on="order_id", how="left")
                   .groupby("customer_unique_id")
                   .agg(total_spent=("order_value","sum"),
                        orders_count=("order_id","nunique"),
                        avg_order_value=("order_value","mean")))

# Recency relative to snapshot_date: days since last purchase (safe because last purchase <= snapshot)
last_purchase = (orders_hist.groupby("customer_unique_id")["order_purchase_timestamp"]
                 .agg(last_purchase_date="max"))
first_purchase = (orders_hist.groupby("customer_unique_id")["order_purchase_timestamp"]
                 .agg(first_purchase_date="min"))

# Inter-purchase intervals (std) for customers with >1 order
orders_sorted = (orders_hist.sort_values(["customer_unique_id","order_purchase_timestamp"])
                 .groupby("customer_unique_id")["order_purchase_timestamp"]
                 .apply(lambda x: x.diff().dt.days.dropna()))
purchase_gap_std = orders_sorted.groupby("customer_unique_id").std().rename("purchase_gap_std")

# Combine features
features = agg_order_value.join(last_purchase).join(first_purchase).join(purchase_gap_std, how="left")
features["recency_days"] = (snapshot_date - features["last_purchase_date"]).dt.days
features["tenure_days"] = (features["last_purchase_date"] - features["first_purchase_date"]).dt.days.fillna(0)

# Fill NaN
features["purchase_gap_std"] = features["purchase_gap_std"].fillna(0)
features["avg_order_value"] = features["avg_order_value"].fillna(0)
features["total_spent"] = features["total_spent"].fillna(0)
features["orders_count"] = features["orders_count"].fillna(0)

# Label: churn if the customer DID NOT purchase in future window
future_customers = orders_future["customer_unique_id"].unique()
features["churn_label"] = (~features.index.isin(future_customers)).astype(int)

# Reset index to have customer id column
features = features.reset_index().rename(columns={"index":"customer_unique_id"})
features.to_csv(r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\data\snapshot_features_churn_labels.csv", index=False)
print("Snapshot features + labels saved to snapshot_features_churn_labels.csv")
print(features[['customer_unique_id','orders_count','total_spent','recency_days','churn_label']].head())
