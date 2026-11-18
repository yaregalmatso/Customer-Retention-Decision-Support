"""
Training & Serialization Script
Matches Google Colab Pipeline: Imputation, Feature Engineering,
Dynamic Class Weighting (scale_pos_weight), and SHAP-ready serialization.
"""

import os
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 5 domain features engineered in the notebook
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


def train_and_export():
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    # Download Telco raw dataset if not present
    data_url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    print(f"Fetching dataset from {data_url}...")
    df = pd.read_csv(data_url)

    # 1. Cleaning & Target Encoding
    df = df.drop("customerID", axis=1, errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # 2. Feature Engineering
    df = engineer_features(df)

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Export sample test set for UI demo loading
    sample_df = X.sample(n=20, random_state=42).copy()
    sample_df.to_csv("data/sample_customers.csv", index=False)
    print("Saved sample profiles to data/sample_customers.csv")

    categorical_cols = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    numerical_cols = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Dynamic Class Weighting
    scale_weight = float((len(y_train) - sum(y_train)) / sum(y_train))
    print(f"Computed scale_pos_weight: {scale_weight:.4f}")

    # Transformation Pipelines
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="missing"),
            ),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    # Production Model
    classifier = xgb.XGBClassifier(
        scale_pos_weight=scale_weight,
        eval_metric="logloss",
        random_state=42,
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
    )

    full_pipeline = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", classifier)]
    )

    print("Fitting model...")
    full_pipeline.fit(X_train, y_train)

    # Extract transformed feature names for SHAP interpretability
    fitted_preprocessor = full_pipeline.named_steps["preprocessor"]
    cat_encoder = fitted_preprocessor.named_transformers_["cat"].named_steps[
        "onehot"
    ]
    encoded_cat_names = cat_encoder.get_feature_names_out(
        categorical_cols
    ).tolist()
    all_feature_names = numerical_cols + encoded_cat_names

    feature_meta = {
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "all_feature_names": all_feature_names,
    }

    joblib.dump(full_pipeline, "models/churn_pipeline.joblib")
    joblib.dump(feature_meta, "models/feature_meta.joblib")
    print("Artifacts saved: models/churn_pipeline.joblib & feature_meta.joblib")


if __name__ == "__main__":
    train_and_export()