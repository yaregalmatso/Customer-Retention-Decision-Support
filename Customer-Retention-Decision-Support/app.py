"""
Streamlit Web Dashboard: Subscription Churn Risk & Retention Optimizer
Course: FUSE AI-201 AI Model Development and Implementation
Group 17 - Florida University Southeast
"""

import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="ChurnGuard AI | Enterprise Retention",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for polished SaaS UI
st.markdown(
    """
<style>
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #e9ecef;
        margin-bottom: 12px;
    }
    .high-risk {
        color: #d90429;
        font-weight: 700;
    }
    .low-risk {
        color: #2b9348;
        font-weight: 700;
    }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_artifacts():
    pipeline_path = "models/churn_pipeline.joblib"
    meta_path = "models/feature_meta.joblib"

    if not os.path.exists(pipeline_path) or not os.path.exists(meta_path):
        st.error(
            "Model artifacts not found! Please run `python train.py` first to generate the trained model."
        )
        st.stop()

    pipeline = joblib.load(pipeline_path)
    meta = joblib.load(meta_path)
    classifier = pipeline.named_steps["classifier"]
    explainer = shap.TreeExplainer(classifier)
    return pipeline, meta, explainer


pipeline, meta, explainer = load_artifacts()


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Replicates the 5 engineered features from the notebook."""
    df = df.copy()
    df["ValueRatio"] = df["TotalCharges"] / (df["tenure"] + 1)
    df["TenureGroup"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 36, 72],
        labels=["New", "Short", "Medium", "Long"],
    )
    df["Monthly_to_Total_Ratio"] = df["MonthlyCharges"] / (
        df["TotalCharges"] + 1
    )
    df["Tenure_Charges_Interaction"] = df["tenure"] * df["MonthlyCharges"]
    df["Total_Charges_to_Monthly_Ratio"] = df["TotalCharges"] / (
        df["MonthlyCharges"] + 1
    )
    return df


def get_retention_prescriptions(top_driver: str, prob: float) -> list:
    """Maps model drivers to concrete business retention actions."""
    actions = []
    if prob < 0.35:
        return [
            "Account health is strong (Low Risk). Suppress discounts to prevent unnecessary revenue leakage.",
            "Candidate for value-add upselling (e.g., streaming entertainment or higher speed tiers).",
        ]

    if "Contract" in top_driver:
        actions.append(
            "Contract Upgrade Offer: Offer an annual rate freeze and a $10/month loyalty credit for switching to a 1-year or 2-year contract."
        )
    if "MonthlyCharges" in top_driver or "ValueRatio" in top_driver:
        actions.append(
            "Pricing Plan Review: Conduct a plan audit to remove unutilized bandwidth or downgrade to a right-sized plan."
        )
    if "PaymentMethod" in top_driver:
        actions.append(
            "Automate Payments: Offer a one-time $15 statement credit for setting up recurring automated bank transfer or credit card payments."
        )
    if "TechSupport" in top_driver or "OnlineSecurity" in top_driver:
        actions.append(
            "Complimentary Protection Bundle: Provide 6 months of free Tech Support and Online Security."
        )
    if "Fiber" in top_driver:
        actions.append(
            "Line Quality Check: Proactively dispatch a technician or run remote line diagnostics to resolve latent fiber service friction."
        )

    if not actions:
        actions.append(
            "Direct Outreach: Assign this account to a dedicated retention specialist for a concierge check-in call."
        )
    return actions


# Header
st.title("🛡️ ChurnGuard AI: Customer Retention Decision Support")
st.caption(
    "Predict subscriber churn risk, evaluate individual driver attributions, and optimize retention interventions."
)

tab1, tab2 = st.tabs(
    ["Single Account Profiler", "Batch Upload & Portfolio Scoring"]
)

# ---------------------------------------------------------
# TAB 1: SINGLE ACCOUNT PROFILER
# ---------------------------------------------------------
with tab1:
    st.sidebar.header("Account Parameters")

    preset = st.sidebar.selectbox(
        "Load Benchmark Preset",
        ["Custom Input", "Edge Case: High Risk", "Edge Case: Low Risk"],
    )

    if preset == "Edge Case: High Risk":
        default_tenure = 2
        default_contract = "Month-to-month"
        default_payment = "Electronic check"
        default_internet = "Fiber optic"
        default_monthly = 85.50
        default_total = 170.00
        default_support = "No"
        default_sec = "No"
        default_paperless = "Yes"
        default_partner = "Yes"
        default_dep = "No"
    elif preset == "Edge Case: Low Risk":
        default_tenure = 65
        default_contract = "Two year"
        default_payment = "Bank transfer (automatic)"
        default_internet = "DSL"
        default_monthly = 89.00
        default_total = 5700.00
        default_support = "Yes"
        default_sec = "Yes"
        default_paperless = "No"
        default_partner = "Yes"
        default_dep = "Yes"
    else:
        default_tenure = 12
        default_contract = "Month-to-month"
        default_payment = "Electronic check"
        default_internet = "Fiber optic"
        default_monthly = 70.00
        default_total = 840.00
        default_support = "No"
        default_sec = "No"
        default_paperless = "Yes"
        default_partner = "No"
        default_dep = "No"

    col_a, col_b, col_c = st.columns(3)

    with col_a:
        st.subheader("Contract & Tenure")
        tenure = st.slider("Tenure (Months)", 0, 72, default_tenure)
        contract = st.selectbox(
            "Contract Term",
            ["Month-to-month", "One year", "Two year"],
            index=["Month-to-month", "One year", "Two year"].index(
                default_contract
            ),
        )
        paperless = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"],
            index=["Yes", "No"].index(default_paperless),
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ],
            index=[
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)",
            ].index(default_payment),
        )

    with col_b:
        st.subheader("Financial & Telephony")
        monthly_charges = st.number_input(
            "Monthly Charges ($)", 18.0, 120.0, float(default_monthly), step=0.5
        )
        total_charges = st.number_input(
            "Total Charges ($)", 0.0, 9000.0, float(default_total), step=10.0
        )
        phone_service = st.selectbox("Phone Service", ["Yes", "No"], index=0)
        mult_lines = st.selectbox(
            "Multiple Lines", ["No", "Yes", "No phone service"], index=0
        )

    with col_c:
        st.subheader("Internet & Services")
        internet = st.selectbox(
            "Internet Service",
            ["Fiber optic", "DSL", "No"],
            index=["Fiber optic", "DSL", "No"].index(default_internet),
        )
        tech_support = st.selectbox(
            "Tech Support",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_support),
        )
        online_sec = st.selectbox(
            "Online Security",
            ["No", "Yes", "No internet service"],
            index=["No", "Yes", "No internet service"].index(default_sec),
        )
        online_backup = st.selectbox(
            "Online Backup", ["No", "Yes", "No internet service"], index=0
        )
        device_prot = st.selectbox(
            "Device Protection", ["No", "Yes", "No internet service"], index=0
        )
        stream_tv = st.selectbox(
            "Streaming TV", ["Yes", "No", "No internet service"], index=0
        )
        stream_movies = st.selectbox(
            "Streaming Movies", ["Yes", "No", "No internet service"], index=0
        )

    with st.expander("Demographics & Household Details"):
        d1, d2, d3, d4 = st.columns(4)
        gender = d1.selectbox("Gender", ["Female", "Male"])
        senior = d2.selectbox("Senior Citizen", [0, 1], index=0)
        partner = d3.selectbox(
            "Partner",
            ["Yes", "No"],
            index=["Yes", "No"].index(default_partner),
        )
        dependents = d4.selectbox(
            "Dependents",
            ["Yes", "No"],
            index=["Yes", "No"].index(default_dep),
        )

    single_dict = {
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone_service,
        "MultipleLines": mult_lines,
        "InternetService": internet,
        "OnlineSecurity": online_sec,
        "OnlineBackup": online_backup,
        "DeviceProtection": device_prot,
        "TechSupport": tech_support,
        "StreamingTV": stream_tv,
        "StreamingMovies": stream_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges,
    }

    raw_input_df = pd.DataFrame([single_dict])
    processed_input_df = engineer_features(raw_input_df)

    if st.button("Evaluate Churn Risk", type="primary", use_container_width=True):
        churn_prob = pipeline.predict_proba(processed_input_df)[0, 1]
        churn_class = pipeline.predict(processed_input_df)[0]

        st.divider()

        m1, m2, m3 = st.columns([1, 1, 2])

        with m1:
            st.metric("Churn Probability", f"{churn_prob * 100:.2f}%")
        with m2:
            if churn_prob >= 0.50:
                st.markdown(
                    '<h3 class="high-risk">⚠️ HIGH CHURN RISK</h3>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<h3 class="low-risk">✅ LOW RISK (RETAINED)</h3>',
                    unsafe_allow_html=True,
                )
        with m3:
            st.progress(
                float(churn_prob),
                text=f"Operational Threshold: 50.0% (Current: {churn_prob*100:.2f}%)",
            )

        # Local SHAP Explanation
        preproc = pipeline.named_steps["preprocessor"]
        clf = pipeline.named_steps["classifier"]
        X_trans = preproc.transform(processed_input_df)
        shap_vals = explainer(X_trans)

        shap_vals.feature_names = meta["all_feature_names"]

        instance_shap = shap_vals[0].values
        top_driver_idx = int(np.argmax(instance_shap))
        top_driver_name = meta["all_feature_names"][top_driver_idx]

        st.subheader("🔍 Local Explainability: Key Churn Drivers")
        col_viz, col_rx = st.columns([3, 2])

        with col_viz:
            fig, ax = plt.subplots(figsize=(8, 4))
            shap.plots.waterfall(shap_vals[0], max_display=7, show=False)
            st.pyplot(fig)
            plt.close()

        with col_rx:
            st.markdown("### 🎯 Recommended Interventions")
            st.markdown(f"**Primary Risk Lever:** `{top_driver_name}`")
            actions = get_retention_prescriptions(top_driver_name, churn_prob)
            for action in actions:
                st.info(action)

# ---------------------------------------------------------
# TAB 2: BATCH PORTFOLIO EVALUATION
# ---------------------------------------------------------
with tab2:
    st.subheader("Batch Portfolio Evaluation")
    st.write(
        "Upload a batch CSV of customer accounts to compute portfolio risk scores, identify churn cohorts, and export retention action lists."
    )

    uploaded_file = st.file_uploader(
        "Upload Customer CSV (Telco schema)", type=["csv"]
    )

    sample_path = "data/prepared_dataset.csv"
    if uploaded_file is None and os.path.exists(sample_path):
        if st.checkbox("Or load built-in benchmark sample dataset (20 accounts)"):
            uploaded_file = sample_path

    if uploaded_file is not None:
        batch_df = (
            pd.read_csv(uploaded_file)
            if isinstance(uploaded_file, str)
            else pd.read_csv(uploaded_file)
        )
        st.write(f"Loaded **{len(batch_df)}** customer records.")

        if st.button("Score Customer Portfolio", type="primary"):
            clean_batch = batch_df.copy()
            if "customerID" in clean_batch.columns:
                cust_ids = clean_batch["customerID"]
                clean_batch = clean_batch.drop(columns=["customerID"])
            else:
                cust_ids = pd.Series(
                    [f"CUST-{i:04d}" for i in range(len(clean_batch))],
                    name="customerID",
                )

            clean_batch["TotalCharges"] = pd.to_numeric(
                clean_batch["TotalCharges"], errors="coerce"
            ).fillna(clean_batch["MonthlyCharges"])

            feat_batch = engineer_features(clean_batch)
            probs = pipeline.predict_proba(feat_batch)[:, 1]

            results_df = batch_df.copy()
            results_df.insert(0, "CustomerID", cust_ids)
            results_df["Churn_Probability"] = np.round(probs * 100, 2)
            results_df["Risk_Tier"] = np.where(
                probs >= 0.65,
                "Critical Risk",
                np.where(probs >= 0.40, "Medium Risk", "Low Risk"),
            )

            # Summary Metrics
            c1, c2, c3 = st.columns(3)
            high_count = int((probs >= 0.50).sum())
            c1.metric("Total Accounts Evaluated", len(results_df))
            c2.metric("Flagged Churn Risk Accounts", high_count)
            c3.metric(
                "Predicted Portfolio Churn Rate",
                f"{(high_count / len(results_df)) * 100:.1f}%",
            )

            st.dataframe(
                results_df[
                    [
                        "CustomerID",
                        "Churn_Probability",
                        "Risk_Tier",
                        "Contract",
                        "tenure",
                        "MonthlyCharges",
                        "PaymentMethod",
                    ]
                ],
                use_container_width=True,
            )

            # Download Action
            csv_data = results_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Scored Churn Portfolio (CSV)",
                data=csv_data,
                file_name="churn_predictions_scored.csv",
                mime="text/csv",
                use_container_width=True,
            )