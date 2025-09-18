# train_models.py
import os
import pandas as pd
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import numpy as np

DB_PATH = "db/kpi_data.db"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

def load_customer_features():
    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    df = pd.read_sql_table("customer_features", engine)
    df.columns = df.columns.map(str)  # force string column names
    return df

def preprocess_for_model(df):
    df = df.copy()

    # Dates
    df["signup_date"] = pd.to_datetime(df["signup_date"])
    if "last_order" in df.columns:
        df["last_order"] = pd.to_datetime(df["last_order"]).fillna(df["signup_date"])
        max_date = df["last_order"].max()
    else:
        max_date = pd.Timestamp.today()
        df["last_order"] = df["signup_date"]

    # Derived features
    df["signup_age_days"] = (pd.to_datetime(max_date) - df["signup_date"]).dt.days
    features = [
        "total_revenue", "orders_count", "avg_order_value",
        "recency_days", "tenure_days", "signup_age_days"
    ]
    df[features] = df[features].fillna(0)

    # Log transforms
    df["total_revenue_log"] = np.log1p(df["total_revenue"])
    df["avg_order_value_log"] = np.log1p(df["avg_order_value"])
    df["orders_count_log"] = np.log1p(df["orders_count"])

    # Feature matrix
    X = df[[
        "total_revenue_log", "orders_count_log", "avg_order_value_log",
        "recency_days", "tenure_days", "signup_age_days"
    ]].copy()
    X.columns = X.columns.map(str)

    # Target variable
    y = df["churn_flag"].astype(int)

    return X, y, df

def train_churn_model(X, y):
    X = pd.DataFrame(X.to_numpy(), columns=X.columns)  # ensure clean DataFrame

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=8, random_state=42, n_jobs=-1
    )
    clf.fit(X_train, y_train)

    preds = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    acc = accuracy_score(y_test, (preds > 0.5).astype(int))
    print(f"✅ Churn model AUC: {auc:.4f}, Accuracy: {acc:.4f}")
    return clf

def train_kmeans(X, n_clusters=6):
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    km.fit(X)
    return km

def main():
    print("Loading customer features...")
    df = load_customer_features()
    X, y, df_full = preprocess_for_model(df)

    print("Training churn model...")
    churn_model = train_churn_model(X, y)
    joblib.dump(churn_model, os.path.join(MODELS_DIR, "churn_model.joblib"))
    print("✅ Saved churn model.")

    print("Training k-means segmentation...")
    X_cluster = X[["total_revenue_log", "orders_count_log", "avg_order_value_log"]].fillna(0)
    kmeans = train_kmeans(X_cluster, n_clusters=6)
    joblib.dump(kmeans, os.path.join(MODELS_DIR, "kmeans_model.joblib"))
    print("✅ Saved kmeans model.")

    # Save predictions back to DB
    df_full["churn_prob"] = churn_model.predict_proba(X)[:, 1]
    df_full["segment"] = kmeans.predict(X_cluster)

    engine = create_engine(f"sqlite:///{DB_PATH}", echo=False)
    df_full[[
        "customer_id", "churn_flag", "churn_prob", "segment",
        "total_revenue", "orders_count", "avg_order_value",
        "recency_days", "tenure_days", "signup_date", "region"
    ]].to_sql("customer_scored", engine, if_exists="replace", index=False)

    print("✅ Saved scored customers to database as 'customer_scored'.")

if __name__ == "__main__":
    main()
