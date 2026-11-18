---
title: ChurnGuard AI Subscription Retention
emoji: 🛡️
colorFrom: red
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🛡️ ChurnGuard AI: Customer Churn Prediction & Retention Optimizer

**Course:** FUSE AI-201 AI Model Development and Implementation
**Institution:** **Florida University Southeast** (School of Professional Studies)
**Project Group:** 17
**Domain:** Retail, E-Commerce, and Telecommunications Subscriptions

ChurnGuard AI is an end-to-end machine learning system and interactive Streamlit web dashboard designed to predict customer churn, explain individual-level churn drivers using SHAP (SHapley Additive exPlanations), and prescribe targeted retention offers before accounts cancel.

---

## 📋 Table of Contents
1. [Project Overview](#-project-overview)
2. [Key Features](#-key-features)
3. [Repository Structure](#-repository-structure)
4. [Prerequisites](#-prerequisites)
5. [Installation](#-installation)
6. [Model Training & Serialization](#-model-training--serialization)
7. [Running the Application](#-running-the-application)
   - [Option 1: Local Execution](#option-1-local-execution)
   - [Option 2: Docker Execution](#option-2-docker-execution)
8. [Using the Dashboard](#-using-the-dashboard)
   - [Single Account Profiler](#1-single-account-profiler)
   - [Batch Portfolio Evaluation](#2-batch-portfolio-evaluation)
9. [Cloud Deployment](#-cloud-deployment)
   - [Deploy to Hugging Face Spaces](#hugging-face-spaces)
   - [Deploy to Render](#render)
10. [Test Benchmarks & Edge Cases](#-test-benchmarks--edge-cases)
11. [Project Contributors](#-project-contributors)

---

## 📖 Project Overview

Customer attrition degrades Monthly Recurring Revenue (MRR) and inflates Customer Acquisition Costs (CAC). This application replaces reactive exit surveys with an **XGBoost Classifier** that:
- Features 5 custom domain-engineered attributes.
- Handles class imbalance using dynamic loss weighting (`scale_pos_weight = 2.77`).
- Achieves **80.21% churn recall** and **0.8469 ROC-AUC** on holdout test data.
- Provides real-time, local prediction explanations via **SHAP Waterfall plots**.

---

## ✨ Key Features

- **Real-Time Risk Scoring:** Calculates continuous churn probabilities and maps them to operational risk tiers (`Low Risk`, `Medium Risk`, `Critical Risk`).
- **Benchmark Presets:** Test high-risk and low-risk edge cases with one click.
- **Explainable AI (XAI):** Visualizes the exact positive and negative feature contributions driving individual customer predictions.
- **Prescriptive Decision Engine:** Maps dominant SHAP risk drivers to business interventions (contract extensions, rate freezes, payment method discounts, free tech support).
- **Batch CSV Scoring:** Upload portfolio data, calculate cohort churn metrics, and export scored customer action lists as CSV files.

---

## 📂 Repository Structure

```text
telecom-churn-app/
├── app.py                      # Streamlit interactive UI application
├── train.py                    # Pipeline execution, preprocessing & model export
├── requirements.txt            # Pinned Python package dependencies
├── Dockerfile                  # Container definition for local, HF Spaces & Render
├── README.md                   # Project documentation & space metadata
├── models/                     # Saved model artifacts (created by train.py)
│   ├── churn_pipeline.joblib   # Serialized preprocessor + XGBoost model
│   └── feature_meta.joblib     # Preprocessor metadata and feature names for SHAP
└── data/
    └── prepared_customer_churn_dataset.csv    # Benchmark dataset for batch testing (created by train.py)
```

---

## ⚙️ Prerequisites

- **Python:** Version `3.10` or higher
- **Package Manager:** `pip`
- **Containerization (Optional):** [Docker Desktop](https://www.docker.com/products/docker-desktop/)

---

## 📥 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Jerusalem-Fetene/Customer-Retention-Decision-Support.git
   ```

2. **Create and activate a virtual environment:**
   - **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     python -m venv venv
     venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

---

## 🏋️ Model Training & Serialization

Before running the application for the first time, generate the model artifacts:

```bash
python train.py
```

### What this script does:
1. Downloads the standard Telco Customer Churn dataset.
2. Cleans whitespace strings in `TotalCharges` and drops `customerID`.
3. Engineers 5 features (`ValueRatio`, `TenureGroup`, `Monthly_to_Total_Ratio`, `Tenure_Charges_Interaction`, `Total_Charges_to_Monthly_Ratio`).
4. Trains an **XGBoost Classifier** with class-imbalance weighting (`scale_pos_weight = 2.77`).
5. Generates the `models/` directory containing `churn_pipeline.joblib` and `feature_meta.joblib`.
6. Generates `data/sample_customers.csv` containing 20 test customer profiles.

---

## 🚀 Running the Application

### Option 1: Local Execution

Run Streamlit directly from your terminal:

```bash
streamlit run app.py
```

Your browser will open automatically at:
```text
http://localhost:8501
```

---

### Option 2: Docker Execution

To build and run using a container:

1. **Build the Docker image:**
   ```bash
   docker build -t churn-app .
   ```

2. **Run the container:**
   ```bash
   docker run -p 7860:7860 churn-app
   ```

3. Navigate to:
   ```text
   http://localhost:7860
   ```

---

## 🖥️ Using the Dashboard

### 1. Single Account Profiler
1. **Choose an Input Source:**
   - Use the **Sidebar Preset** to select `Custom Input`, `Edge Case: High Risk`, or `Edge Case: Low Risk`.
2. **Configure Parameters:**
   - Modify Tenure, Contract type, Payment method, Monthly/Total charges, and add-on services.
3. **Click `Evaluate Churn Risk`:**
   - **Risk Gauge:** Displays predicted churn probability and category badge (`HIGH CHURN RISK` vs. `LOW RISK`).
   - **SHAP Waterfall Plot:** Displays individual positive drivers (pushing toward churn) and negative drivers (pushing toward retention).
   - **Targeted Interventions:** Suggests business actions matching the primary churn driver.

### 2. Batch Portfolio Evaluation
1. Switch to the **Batch Upload & Portfolio Scoring** tab.
2. Check the box to use the included benchmark dataset, or upload any `.csv` file matching the Telco dataset schema.
3. Click **`Score Customer Portfolio`** to compute:
   - Flagged accounts and predicted portfolio churn rate.
   - A filterable table with individual risk probabilities and tiers.
4. Click **`📥 Download Scored Churn Portfolio (CSV)`** to export the results.

---

## ☁️ Cloud Deployment

### Hugging Face Spaces
1. Create a new Space at [huggingface.co/spaces](https://huggingface.co/spaces).
2. Set the Space SDK to **Docker** (Blank template).
3. Clone the repo and push all project files:
   ```bash
   git add .
   git commit -m "Deploy ChurnGuard AI"
   git push
   ```
4. Hugging Face Spaces reads the YAML metadata in this `README.md`, executes the `Dockerfile`, and serves the app on port `7860`.

### Render
1. Push your repository to GitHub.
2. Sign in to [render.com](https://render.com) and click **New +** -> **Web Service**.
3. Connect your repository and select the **Docker** runtime.
4. Click **Deploy Web Service**. Render reads the `$PORT` variable configured in the `Dockerfile` and deploys to a public HTTPS endpoint.

---

## 🧪 Test Benchmarks & Edge Cases

| Attribute | High-Risk Benchmark (~87% Churn) | Low-Risk Benchmark (~7% Churn) |
| :--- | :--- | :--- |
| **Tenure** | `2 Months` | `65 Months` |
| **Contract** | `Month-to-month` | `Two year` |
| **Internet Service** | `Fiber optic` | `DSL` |
| **Tech Support** | `No` | `Yes` |
| **Online Security** | `No` | `Yes` |
| **Payment Method** | `Electronic check` | `Bank transfer (automatic)` |
| **Monthly Charges** | `$85.50` | `$89.00` |
| **Total Charges** | `$170.00` | `$5,700.00` |

---

## 👥 Project Contributors

- **Biniyam Luelseged Mojo**
- **Jerusalem Fetene Mekonnin**
- **Misgana Aragaw Mekuria**
- **Sahale Shera Lutse**
- **Teshome Talema Mekonnen**
- **Yaregal Matso Mawura**
- **Yaregal Wubalem Liku**