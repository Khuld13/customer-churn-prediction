import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Churn Predictor", page_icon="📡", layout="centered")
st.title("📡 Customer Churn Predictor")
st.markdown("Fill in the customer details below and click **Predict** to see the churn risk.")

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("churn_model_final.pkl")

model = load_model()

# ── Input form ─────────────────────────────────────────────────────────────────
st.subheader("Customer Details")

col1, col2 = st.columns(2)

with col1:
    tenure         = st.slider("Tenure (Months)", 0, 72, 12)
    monthly        = st.number_input("Monthly Charges ($)", 0.0, 200.0, 65.0, step=0.5)
    total          = st.number_input("Total Charges ($)", 0.0, 10000.0, float(tenure * 65), step=1.0)
    gender         = st.selectbox("Gender", ["Male", "Female"])
    senior         = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner        = st.selectbox("Partner", ["No", "Yes"])
    dependents     = st.selectbox("Dependents", ["No", "Yes"])
    phone_service  = st.selectbox("Phone Service", ["No", "Yes"])
    paperless      = st.selectbox("Paperless Billing", ["No", "Yes"])

with col2:
    multiple_lines = st.selectbox("Multiple Lines",       ["No", "Yes", "No phone service"])
    internet       = st.selectbox("Internet Service",     ["DSL", "Fiber optic", "No"])
    online_sec     = st.selectbox("Online Security",      ["No", "Yes", "No internet service"])
    online_bkp     = st.selectbox("Online Backup",        ["No", "Yes", "No internet service"])
    device_prot    = st.selectbox("Device Protection",    ["No", "Yes", "No internet service"])
    tech_support   = st.selectbox("Tech Support",         ["No", "Yes", "No internet service"])
    streaming_tv   = st.selectbox("Streaming TV",         ["No", "Yes", "No internet service"])
    streaming_mov  = st.selectbox("Streaming Movies",     ["No", "Yes", "No internet service"])
    contract       = st.selectbox("Contract",             ["Month-to-month", "One year", "Two year"])
    payment        = st.selectbox("Payment Method",       ["Electronic check", "Mailed check",
                                                            "Bank transfer (automatic)", "Credit card (automatic)"])

# City frequency — use median frequency (neutral stand-in for unknown city)
city_freq = 0.003   # ~median frequency across 1,100 cities in training data

# ── Encode inputs to match training pipeline ───────────────────────────────────
def build_input():
    row = {
        # Numeric
        "City":             city_freq,
        "Zip Code":         0,          # dropped in model; kept as placeholder
        "Latitude":         0.0,
        "Longitude":        0.0,
        "Tenure Months":    tenure,
        "Monthly Charges":  monthly,
        "Total Charges":    total,

        # Binary 0/1
        "Gender":           1 if gender == "Male" else 0,
        "Senior Citizen":   1 if senior == "Yes" else 0,
        "Partner":          1 if partner == "Yes" else 0,
        "Dependents":       1 if dependents == "Yes" else 0,
        "Phone Service":    1 if phone_service == "Yes" else 0,
        "Paperless Billing":1 if paperless == "Yes" else 0,

        # One-hot: Multiple Lines (drop_first → ref = "No")
        "Multiple Lines_No phone service": 1 if multiple_lines == "No phone service" else 0,
        "Multiple Lines_Yes":              1 if multiple_lines == "Yes" else 0,

        # One-hot: Internet Service (ref = "DSL")
        "Internet Service_Fiber optic":    1 if internet == "Fiber optic" else 0,
        "Internet Service_No":             1 if internet == "No" else 0,

        # One-hot: Online Security (ref = "No")
        "Online Security_No internet service": 1 if online_sec == "No internet service" else 0,
        "Online Security_Yes":                 1 if online_sec == "Yes" else 0,

        # One-hot: Online Backup (ref = "No")
        "Online Backup_No internet service":   1 if online_bkp == "No internet service" else 0,
        "Online Backup_Yes":                   1 if online_bkp == "Yes" else 0,

        # One-hot: Device Protection (ref = "No")
        "Device Protection_No internet service": 1 if device_prot == "No internet service" else 0,
        "Device Protection_Yes":                 1 if device_prot == "Yes" else 0,

        # One-hot: Tech Support (ref = "No")
        "Tech Support_No internet service":    1 if tech_support == "No internet service" else 0,
        "Tech Support_Yes":                    1 if tech_support == "Yes" else 0,

        # One-hot: Streaming TV (ref = "No")
        "Streaming TV_No internet service":    1 if streaming_tv == "No internet service" else 0,
        "Streaming TV_Yes":                    1 if streaming_tv == "Yes" else 0,

        # One-hot: Streaming Movies (ref = "No")
        "Streaming Movies_No internet service": 1 if streaming_mov == "No internet service" else 0,
        "Streaming Movies_Yes":                 1 if streaming_mov == "Yes" else 0,

        # One-hot: Contract (ref = "Month-to-month")
        "Contract_One year":  1 if contract == "One year" else 0,
        "Contract_Two year":  1 if contract == "Two year" else 0,

        # One-hot: Payment Method (ref = "Bank transfer (automatic)")
        "Payment Method_Credit card (automatic)": 1 if payment == "Credit card (automatic)" else 0,
        "Payment Method_Electronic check":        1 if payment == "Electronic check" else 0,
        "Payment Method_Mailed check":            1 if payment == "Mailed check" else 0,
    }
    return pd.DataFrame([row])


# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🔍 Predict Churn"):
    input_df = build_input()

    # Align to training columns (handle any mismatch gracefully)
    try:
        train_cols = model.feature_names_in_
        input_df = input_df.reindex(columns=train_cols, fill_value=0)
    except AttributeError:
        pass  # older sklearn; skip alignment

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.markdown("---")
    st.subheader("Prediction Result")

    if prediction == 1:
        st.error(f"⚠️ **High Churn Risk** — {probability:.1%} probability of churning")
    else:
        st.success(f"✅ **Low Churn Risk** — {probability:.1%} probability of churning")

    # ── Feature importance chart ───────────────────────────────────────────────
    st.subheader("Top 10 Features Driving Churn")

    importances = pd.Series(model.feature_importances_, index=model.feature_names_in_)
    top10 = importances.sort_values(ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(7, 4))
    top10[::-1].plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features (Random Forest)")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

st.markdown("---")
st.caption("Model: Random Forest (tuned) · F1 = 0.643 · AUC-ROC = 0.853 · IBM Telco Dataset")
