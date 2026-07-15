# 🧠 Customer Churn Intelligence Platform

## Overview

Customer churn is one of the biggest challenges faced by subscription-based businesses. Losing customers directly impacts revenue, customer acquisition costs, and long-term growth.

The **Customer Churn Intelligence Platform** is an end-to-end Machine Learning application designed to predict customers at risk of leaving and provide actionable insights to support retention strategies.

The project demonstrates a complete ML workflow:

- Data exploration
- Data preprocessing
- Feature engineering
- Model development
- Model comparison
- Hyperparameter optimization
- Model interpretation
- Deployment using Streamlit

---

# Business Problem

Customer acquisition is expensive. Identifying customers who are likely to churn allows businesses to intervene early through targeted retention strategies.

The objective of this project is to develop a predictive model that answers:

> "Which customers are most likely to churn, and why?"

---

# Dataset

Dataset:

**IBM Telco Customer Churn Dataset**

The dataset contains customer demographic information, service subscriptions, account details, and churn status.

Features include:

- Customer tenure
- Contract type
- Monthly charges
- Internet services
- Payment methods
- Support services
- Customer demographics


---

# Machine Learning Workflow

## 1. Exploratory Data Analysis

Performed analysis on:

- Customer distribution
- Churn rate
- Contract behaviour
- Tenure patterns
- Service usage
- Pricing patterns


## 2. Data Preprocessing

Implemented:

- Missing value handling
- Numerical feature processing
- Categorical encoding
- Feature transformation

using:


---

# Model Development

Several classification algorithms were evaluated:

| Model | Accuracy | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 80.4% | 0.836 |
| Decision Tree | 79.0% | 0.828 |
| Random Forest | 79.2% | 0.834 |
| XGBoost | 77.9% | 0.823 |

---

# Final Model

The selected model:

## Logistic Regression

Reason:

- Highest ROC-AUC performance
- Strong interpretability
- Suitable for business decision-making

Performance:

| Metric | Score |
|---|---:|
| Accuracy | 80.4% |
| Precision | 64.8% |
| Recall | 57.2% |
| F1 Score | 60.8% |
| ROC-AUC | 83.6% |

---

# Key Business Insights

The model identified the strongest churn drivers:

### 1. Customer Tenure

Customers with shorter relationships are significantly more likely to churn.

### 2. Contract Type

Month-to-month customers have substantially higher churn risk.

### 3. Pricing

Higher monthly charges contribute to increased churn probability.

### 4. Support Services

Customers without additional support services show increased churn tendency.

---

# Application Features

## Dashboard

Provides:

- Model performance summary
- Churn insights
- Business recommendations


## Single Prediction

Allows users to upload customer information and receive:

- Churn prediction
- Probability score
- Risk classification


## Batch Prediction

Users can:

- Upload multiple customers
- Generate predictions
- Download results


## Model Insights

Displays:

- Performance metrics
- Important predictors
- Business interpretation

---

# Technology Stack

## Programming

- Python

## Machine Learning

- Scikit-Learn
- XGBoost

## Data Analysis

- Pandas
- NumPy

## Visualization

- Plotly

## Deployment

- Streamlit


---

# Project Structure
customer-churn-intelligence/

│
├── app.py
├── churn_prediction_pipeline.pkl
├── requirements.txt
├── README.md
│
└── .streamlit/
└── config.toml


---

# Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```


Future Improvements

Planned enhancements:

SHAP explainability
Automated retention recommendations
Cloud deployment
Real-time customer scoring API
CRM integration

