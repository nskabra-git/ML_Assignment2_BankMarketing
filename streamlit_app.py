import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix

st.set_page_config(page_title="Bank Marketing App")

st.title("Bank Marketing Classification App")

# -----------------------------
# Model Selection FIRST
# -----------------------------
model_options = {
    "Logistic Regression": "model/logistic_regression_model.pkl",
    "Decision Tree": "model/decision_tree_model.pkl",
    "KNN": "model/knn_model.pkl",
    "Naive Bayes": "model/naive_bayes_model.pkl",
    "Random Forest": "model/random_forest_model.pkl",
    "XGBoost": "model/xgboost_model.pkl"
}

sample_file_path = "model/sample_test_data.csv"

left_col, right_col = st.columns([1, 2])
with left_col:
    # st.header("User Controls")

    # Model Selection
    selected_model_name = st.selectbox(
        "Select Model",
        list(model_options.keys())
    )

    # Download Section
    if os.path.exists(sample_file_path):
        with open(sample_file_path, "rb") as file:
            st.download_button(
                label="Download Sample Test Dataset",
                data=file,
                file_name="sample_test_data.csv",
                mime="text/csv",
                key="download_sample_data"
            )
    else:
        st.warning("Sample test dataset not found.")

    # -----------------------------
    # Upload Test Dataset
    # -----------------------------
    st.subheader("Upload Test Dataset")
    uploaded_file = st.file_uploader("Upload Test Dataset (CSV)", type=["csv"])


with right_col:

    # st.header("Model Evaluation & Results")

    # Default Evaluation Code Here

# selected_model_name = st.selectbox("Select Model", list(model_options.keys()))

    # -----------------------------
    # Use Default Model for Evaluation - Default Evaluation Section
    # -----------------------------
    # st.subheader("Default Model Evaluation (Sample Test Dataset)")
    st.subheader("Default Model - Evaluation Metrics")

    default_data_path = "model/sample_test_with_target.csv"

    if os.path.exists(default_data_path):

        default_data = pd.read_csv(default_data_path)

        y_true_default = default_data["y"]
        X_default = default_data.drop("y", axis=1)

        default_model_path = model_options[selected_model_name]

        with open(default_model_path, "rb") as f:
            default_model = pickle.load(f)

        y_pred_default = default_model.predict(X_default)

        from sklearn.metrics import (
            accuracy_score,
            roc_auc_score,
            precision_score,
            recall_score,
            f1_score,
            matthews_corrcoef,
            confusion_matrix,
            classification_report
        )

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(accuracy_score(y_true_default, y_pred_default), 4))
        col2.metric("Precision", round(precision_score(y_true_default, y_pred_default), 4))
        col3.metric("Recall", round(recall_score(y_true_default, y_pred_default), 4))

        col1.metric("F1 Score", round(f1_score(y_true_default, y_pred_default), 4))
        col2.metric("MCC", round(matthews_corrcoef(y_true_default, y_pred_default), 4))

        if hasattr(default_model, "predict_proba"):
            y_prob_default = default_model.predict_proba(X_default)[:, 1]
            col3.metric("AUC", round(roc_auc_score(y_true_default, y_prob_default), 4))

        import matplotlib.pyplot as plt
        import seaborn as sns
       # st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_true_default, y_pred_default)

        fig, ax = plt.subplots()
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")

        st.pyplot(fig)
        # st.write(confusion_matrix(y_true_default, y_pred_default))

        # st.subheader("Classification Report")
        # st.text(classification_report(y_true_default, y_pred_default))
        from sklearn.metrics import classification_report

        st.subheader("Classification Report")

        report_dict = classification_report(
            y_true_default,
            y_pred_default,
            output_dict=True
        )

        report_df = pd.DataFrame(report_dict).transpose()

        st.dataframe(report_df)

# -----------------------------
# Run Prediction
# -----------------------------
if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(data.head())

    if st.button("Run Prediction"):

        model_path = model_options[selected_model_name]

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        # Separate target if present
        if "y" in data.columns:
            y_true = data["y"]
            X_input = data.drop("y", axis=1)
        else:
            y_true = None
            X_input = data

        predictions = model.predict(X_input)

        # Create full results dataframe
        results_df = X_input.copy()
        results_df["Prediction"] = predictions

        st.subheader("Prediction Preview (First 10 Rows)")
        st.dataframe(results_df.head(10))
        # st.write(predictions[:10])

        # Download full predictions
        csv = results_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            label="Download Full Predictions",
            data=csv,
            file_name="predictions.csv",
            mime="text/csv",
            key="download_predictions"
        )

        # If target exists → show evaluation
        if y_true is not None:
            st.subheader("Classification Report")
            st.text(classification_report(y_true, predictions))

            st.subheader("Confusion Matrix")
            st.write(confusion_matrix(y_true, predictions))
