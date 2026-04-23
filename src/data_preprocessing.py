import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import yaml
import os


def load_config(config_path: str = "../configs/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Dataset yuklandi: {df.shape[0]} satr, {df.shape[1]} ustun")
    return df


def preprocess(df: pd.DataFrame, config: dict):
    numeric_cols = config["features"]["numeric"]
    categorical_cols = config["features"]["categorical"]
    target = config["features"]["target"]

    # Faqat kerakli ustunlarni olish
    cols = numeric_cols + categorical_cols + [target]
    df = df[cols].copy()

    # Raqamli ustunlar uchun median bilan to'ldirish
    num_imputer = SimpleImputer(strategy="median")
    df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])

    # Kategorik ustunlar uchun most_frequent bilan to'ldirish
    cat_imputer = SimpleImputer(strategy="most_frequent")
    df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

    # Label encoding
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=[target])
    y = df[target]

    return X, y


def split_and_scale(X, y, config: dict):
    test_size = config["data"]["test_size"]
    random_state = config["data"]["random_state"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler


def get_data_stats(df: pd.DataFrame) -> dict:
    return {
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "missing_values": int(df.isnull().sum().sum()),
        "survival_rate": float(df["Survived"].mean()),
    }
