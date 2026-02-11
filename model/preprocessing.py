import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def load_and_split_data(file_path):
    """
    Loads dataset, encodes target, and performs stratified train-test split.
    """

    # Load dataset
    df = pd.read_csv(file_path, sep=";")

    # Encode target variable
    df['y'] = df['y'].map({'no': 0, 'yes': 1})

    # Separate features and target
    X = df.drop('y', axis=1)
    y = df['y']

    # Stratified split to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(X):
    """
    Creates preprocessing pipeline for numerical and categorical features.
    """

    # Identify column types
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_
