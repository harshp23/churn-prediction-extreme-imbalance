import streamlit as st
import pandas as pd
import joblib
import numpy as np

# -------------------------------
# Load model and feature list
# -------------------------------
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\churn_xgb_pipeline.joblib")
    feature_cols = joblib.load(r"C:\Users\harsh\OneDrive\Desktop\data analysis\Agentic AI-Powered Revenue & Churn Analytics on E-Commerce Data\churn_feature_cols.joblib")
    return model, feature_cols

model, feature_cols = load_model()

# Streamlit UI

st.title("📊 Customer Churn Prediction App (XGBoost)")

st.markdown("""
Upload a CSV file with customer features to predict **churn probability**.
The app will calculate probability and predicted churn using a tuned threshold.
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])


threshold = st.slider(
    "Set Churn Threshold (Probability cutoff for predicting churn)",
    min_value=0.0, max_value=1.0, value=0.0267, step=0.001
)

if uploaded_file is not None:
    try:
        # Read uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Check required features
        missing_cols = [col for col in feature_cols if col not in df.columns]
        if missing_cols:
            st.error(f"❌ Missing columns: {missing_cols}")
        else:
            # Ensure correct feature order and fill missing values
            X = df[feature_cols].fillna(0)

            # Predict probabilities
            probs = model.predict_proba(X)[:, 1]
            preds = (probs >= threshold).astype(int)

            # Add results to dataframe
            df["Churn_Probability"] = probs
            df["Predicted_Churn"] = preds

            st.success("✅ Predictions completed!")
            st.dataframe(df.head(10))

            # Option to download results
            csv_out = df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Predictions",
                data=csv_out,
                file_name="churn_predictions.csv",
                mime="text/csv",
            )

            # Optional: Summary statistics
            st.markdown("### Prediction Summary")
            st.write(df["Predicted_Churn"].value_counts())
            st.bar_chart(df["Predicted_Churn"].value_counts())

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")
