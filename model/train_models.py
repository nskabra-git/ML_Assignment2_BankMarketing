import os
import pickle
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier

from preprocessing import load_and_split_data, create_preprocessor
from evaluate import evaluate_model


def main():

    # File path
    data_path = "../data/bank-full.csv"

    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data(data_path)

    # Create preprocessor
    preprocessor = create_preprocessor(X_train)

    # Dictionary of models
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    results = {}

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        # Create pipeline
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        # Train
        pipeline.fit(X_train, y_train)

        # Evaluate
        metrics = evaluate_model(pipeline, X_test, y_test)

        results[model_name] = metrics

        # Save model
        filename = f"{model_name.lower().replace(' ', '_')}_model.pkl"
        filepath = os.path.join(".", filename)

        with open(filepath, "wb") as f:
            pickle.dump(pipeline, f)

    # Create comparison dataframe
    results_df = pd.DataFrame(results).T

    print("\nModel Comparison:")
    print(results_df)


if __name__ == "__main__":
    main()
