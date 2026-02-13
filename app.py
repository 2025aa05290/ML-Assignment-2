import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

st.set_page_config(page_title="Heart Disease ML App", layout="wide")

st.title("Heart Disease Classification - ML Models")

# Load scaler
scaler = joblib.load("model/scaler.pkl")

# Model selection
model_option = st.selectbox(
    "Select Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

model_filename = model_option.replace(" ", "_") + ".pkl"
model = joblib.load(f"model/{model_filename}")

# Upload CSV
uploaded_file = st.file_uploader("Upload Test CSV File", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.write("Preview of Uploaded Data:")
    st.dataframe(df.head())

    if "target" in df.columns:
        y_true = df["target"]
        X = df.drop("target", axis=1)
    else:
        st.error("CSV must contain 'target' column.")
        st.stop()

    # Scaling for required models
    if model_option in ["Logistic Regression", "KNN"]:
        X = scaler.transform(X)

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    st.subheader("Evaluation Metrics")

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_true, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_true, y_pred), 4))
    col3.metric("Recall", round(recall_score(y_true, y_pred), 4))

    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y_true, y_pred), 4))
    col5.metric("AUC", round(roc_auc_score(y_true, y_prob), 4))
    col6.metric("MCC", round(matthews_corrcoef(y_true, y_pred), 4))

    # Confusion Matrix
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Classification Report
    st.subheader("Classification Report")
    st.text(classification_report(y_true, y_pred))

