import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px


# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Churn Intelligence Platform",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==========================
# CUSTOM CSS
# ==========================

st.markdown(
"""
<style>

.main {
    background-color:#F8FAFC;
}

section[data-testid="stSidebar"] {
    width:320px !important;
    background-color:#E3F2FD;
}

.stMetric {
    background-color:white;
    padding:15px;
    border-radius:12px;
    box-shadow:0px 2px 8px rgba(0,0,0,0.08);
}

div[data-testid="stButton"] button {
    width:100%;
    border-radius:8px;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.08);
}

.footer {
    text-align:center;
    color:#64748B;
    font-size:14px;
    margin-top:50px;
}

</style>
""",
unsafe_allow_html=True
)


# ==========================
# LOAD MODEL
# ==========================

@st.cache_resource
def load_model():
    return joblib.load("churn_prediction_pipeline.pkl")


model = load_model()


# ==========================
# SIDEBAR
# ==========================

st.sidebar.title("🧠 Churn Intelligence")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Single Prediction",
        "Batch Prediction",
        "Model Insights",
        "About"
    ]
)


# ==========================
# FOOTER FUNCTION
# ==========================

def footer():

    st.markdown("---")

    st.markdown(
    """
    <div class="footer">

    Built by <b>Badawi Aminu Muhammed</b><br>
    AI / Machine Learning Engineer<br>
    Customer Churn Intelligence Platform © 2026

    </div>
    """,
    unsafe_allow_html=True
    )


# ==========================
# DASHBOARD
# ==========================

if page == "Dashboard":

    st.title("Customer Churn Intelligence Platform")

    st.write(
        """
        An end-to-end Machine Learning platform that predicts customer churn
        and transforms predictions into actionable retention strategies.
        """
    )


    col1,col2,col3,col4 = st.columns(4)

    col1.metric(
        "Model",
        "Logistic Regression"
    )

    col2.metric(
        "Accuracy",
        "80.4%"
    )

    col3.metric(
        "ROC-AUC",
        "0.836"
    )

    col4.metric(
        "Features",
        "41"
    )


    st.subheader("Business Insights")


    insights = pd.DataFrame(
        {
            "Factor":[
                "Low Tenure",
                "Month-to-month Contract",
                "High Charges",
                "No Tech Support"
            ],

            "Impact":[
                "High Risk",
                "High Risk",
                "Medium Risk",
                "Medium Risk"
            ]
        }
    )


    fig = px.bar(
        insights,
        x="Factor",
        y="Impact",
        title="Major Churn Drivers"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.info(
        """
        Key finding:
        
        Customers with short tenure and flexible month-to-month contracts
        represent the highest churn risk.
        """
    )


# ==========================
# SINGLE PREDICTION
# ==========================

elif page == "Single Prediction":


    st.title("Single Customer Churn Prediction")

    st.info("""
        This page predicts churn for **one customer**.

        Upload a CSV file containing **one row only** with the following variables:

        gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService,
        OnlineSecurity, OnlineBackup, DeviceProtection,
        TechSupport, StreamingTV, StreamingMovies,
        Contract, PaperlessBilling, PaymentMethod,
        MonthlyCharges and TotalCharges.

        Do not include **customerID** or **Churn**.
        """)
    
    st.write(
        "Enter customer information to estimate churn probability."
    )

    sample = pd.DataFrame({
        "gender":["Female"],
        "SeniorCitizen":[0],
        "Partner":["Yes"],
        "Dependents":["No"],
        "tenure":[12],
        "PhoneService":["Yes"],
        "MultipleLines":["No"],
        "InternetService":["Fiber optic"],
        "OnlineSecurity":["No"],
        "OnlineBackup":["Yes"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["Yes"],
        "StreamingMovies":["No"],
        "Contract":["Month-to-month"],
        "PaperlessBilling":["Yes"],
        "PaymentMethod":["Electronic check"],
        "MonthlyCharges":[75.25],
        "TotalCharges":[903.00]
    })

    csv = sample.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample CSV",
        csv,
        "sample_customer_data.csv",
        "text/csv"
    )

    st.divider()
    
    uploaded = st.file_uploader(
        "Upload a single customer CSV",
        type=["csv"]
    )


    if uploaded:

        customer = pd.read_csv(uploaded)


        st.dataframe(customer)


        if st.button("Predict Churn"):


            prediction = model.predict(customer)

            probability = model.predict_proba(customer)[0][1]


            risk = (
                "High Risk"
                if probability > 0.7
                else
                "Medium Risk"
                if probability > 0.4
                else
                "Low Risk"
            )


            c1,c2,c3 = st.columns(3)


            c1.metric(
                "Prediction",
                prediction[0]
            )

            c2.metric(
                "Churn Probability",
                f"{probability:.1%}"
            )

            c3.metric(
                "Risk Level",
                risk
            )


            if risk == "High Risk":

                st.error(
                    "Recommended action: Offer retention incentives and proactive support."
                )

            elif risk == "Medium Risk":

                st.warning(
                    "Recommended action: Monitor engagement and offer loyalty benefits."
                )

            else:

                st.success(
                    "Customer appears stable."
                )


# ==========================
# BATCH PREDICTION
# ==========================

elif page == "Batch Prediction":

    st.title("📂 Batch Prediction")

    st.markdown("""
    ### Upload Customer Dataset

    Upload a **CSV file** containing customer information to generate churn predictions.

    **Required columns**

    - gender
    - SeniorCitizen
    - Partner
    - Dependents
    - tenure
    - PhoneService
    - MultipleLines
    - InternetService
    - OnlineSecurity
    - OnlineBackup
    - DeviceProtection
    - TechSupport
    - StreamingTV
    - StreamingMovies
    - Contract
    - PaperlessBilling
    - PaymentMethod
    - MonthlyCharges
    - TotalCharges

    **Do NOT include**

    - customerID
    - Churn

    The uploaded dataset must have the same structure used to train the Machine Learning model.
    """)

    # -----------------------------
    # Sample CSV
    # -----------------------------

    sample = pd.DataFrame({
        "gender":["Female"],
        "SeniorCitizen":[0],
        "Partner":["Yes"],
        "Dependents":["No"],
        "tenure":[12],
        "PhoneService":["Yes"],
        "MultipleLines":["No"],
        "InternetService":["Fiber optic"],
        "OnlineSecurity":["No"],
        "OnlineBackup":["Yes"],
        "DeviceProtection":["No"],
        "TechSupport":["No"],
        "StreamingTV":["Yes"],
        "StreamingMovies":["No"],
        "Contract":["Month-to-month"],
        "PaperlessBilling":["Yes"],
        "PaymentMethod":["Electronic check"],
        "MonthlyCharges":[75.25],
        "TotalCharges":[903.00]
    })

    csv = sample.to_csv(index=False).encode("utf-8")

    st.download_button(
        "📥 Download Sample CSV",
        csv,
        "sample_customer_data.csv",
        "text/csv"
    )

    st.divider()

    uploaded_file = st.file_uploader(
        "Upload Customer CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        st.success("Dataset uploaded successfully.")

        st.subheader("Preview")

        st.dataframe(df.head())

        if st.button("🚀 Predict Churn"):

            predictions = model.predict(df)

            probabilities = model.predict_proba(df)[:,1]

            df["Prediction"] = predictions

            df["Churn Probability"] = probabilities.round(3)

            df["Risk"] = np.where(
                probabilities > 0.70,
                "High",
                np.where(
                    probabilities > 0.40,
                    "Medium",
                    "Low"
                )
            )

            st.success("Prediction completed successfully.")

            st.dataframe(df)

            output = df.to_csv(index=False).encode("utf-8")

            st.download_button(
                "📥 Download Prediction Results",
                output,
                "customer_churn_predictions.csv",
                "text/csv"
            )

# ==========================
# MODEL INSIGHTS
# ==========================

elif page == "Model Insights":


    st.title("Model Performance")


    metrics = pd.DataFrame(
        {
            "Metric":[
                "Accuracy",
                "Precision",
                "Recall",
                "F1 Score",
                "ROC-AUC"
            ],

            "Value":[
                .804,
                .648,
                .572,
                .608,
                .836
            ]
        }
    )


    fig = px.bar(
        metrics,
        x="Metric",
        y="Value",
        title="Model Evaluation Metrics"
    )


    st.plotly_chart(
        fig,
        use_container_width=True
    )


    st.subheader(
        "Important Predictors"
    )


    feature_importance = pd.DataFrame(
        {
            "Feature":[
                "Tenure",
                "Contract Type",
                "Total Charges",
                "Internet Service",
                "Monthly Charges"
            ],

            "Importance":[
                1.35,
                .78,
                .64,
                .61,
                .54
            ]
        }
    )


    fig2 = px.bar(
        feature_importance,
        x="Importance",
        y="Feature",
        orientation="h"
    )


    st.plotly_chart(
        fig2,
        use_container_width=True
    )


# ==========================
# ABOUT
# ==========================

elif page == "About":


    st.title("About")


    st.markdown(
    """
    ## Customer Churn Intelligence Platform

    This project demonstrates a complete Machine Learning workflow:

    - Exploratory Data Analysis
    - Feature Engineering
    - Model Development
    - Model Comparison
    - Hyperparameter Optimization
    - Business Interpretation
    - Deployment

    **Technology Stack**

    - Python
    - Scikit-Learn
    - XGBoost
    - Streamlit
    - Plotly
    - Pandas

    """
    )


footer()