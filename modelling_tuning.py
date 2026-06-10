import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score, ConfusionMatrixDisplay
)
import sklearn.utils
import mlflow
import mlflow.sklearn

# Set experiment name
EXPERIMENT_NAME = "Latihan Credit Scoring"

# Initialize MLflow tracking
dagshub_owner = os.getenv("DAGSHUB_REPO_OWNER")
dagshub_repo = os.getenv("DAGSHUB_REPO_NAME")
dagshub_token = os.getenv("DAGSHUB_TOKEN")

if dagshub_owner and dagshub_repo and dagshub_token:
    print(f"Initializing DagsHub MLflow tracking for {dagshub_owner}/{dagshub_repo}...")
    # Pakai token langsung, tanpa OAuth
    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    mlflow.set_tracking_uri(f"https://dagshub.com/{dagshub_owner}/{dagshub_repo}.mlflow")
    mlflow.set_experiment(EXPERIMENT_NAME)
    print("DagsHub MLflow tracking initialized successfully.")
else:
    print("No DagsHub credentials found. Using local tracking URI (http://127.0.0.1:5000)")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)


def find_dataset():
    """Cari dataset sesuai struktur Dicoding: preprocessing/namadataset_preprocessing/"""
    filename = "credit_scoring_preprocessing.csv"
    script_dir = os.path.dirname(os.path.abspath(__file__))

    candidates = [
        os.path.join("preprocessing", "namadataset_preprocessing", filename),
        os.path.join(script_dir, "preprocessing", "namadataset_preprocessing", filename),
    ]
    for path in candidates:
        if os.path.exists(path):
            print(f"Dataset ditemukan di: {os.path.abspath(path)}")
            return path
    raise FileNotFoundError(
        f"File '{filename}' tidak ditemukan.\n"
        "Pastikan preprocessing sudah dijalankan.\n"
        f"Path yang dicoba:\n" + "\n".join(f"  - {os.path.abspath(p)}" for p in candidates)
    )

def main():
    data_path = find_dataset()
    df = pd.read_csv(data_path)
    print(f"Loaded preprocessed data of shape {df.shape}")

    # Drop kolom non-numerik yang tidak diperlukan
    cols_to_drop = ['default']
    if 'customer_id' in df.columns:
        cols_to_drop.append('customer_id')
    X = df.drop(columns=cols_to_drop)
    y = df['default']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

        run_id = os.getenv("MLFLOW_RUN_ID")
        with mlflow.start_run(run_id=run_id) as run:
        print(f"Started MLflow run: {run.info.run_id}")

        rf = RandomForestClassifier(random_state=42)
        param_grid = {
            'n_estimators': [50, 100],
            'max_depth': [5, 10],
            'min_samples_split': [2, 5]
        }

        print("Starting GridSearchCV for hyperparameter tuning...")
        grid_search = GridSearchCV(
            estimator=rf, param_grid=param_grid, cv=3, scoring='f1', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        print(f"Best Hyperparameters: {best_params}")

        for param_name, param_value in best_params.items():
            mlflow.log_param(param_name, param_value)

        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_pred_proba)
        }
        print(f"Model Metrics: {metrics}")

        for metric_name, metric_value in metrics.items():
            mlflow.log_metric(metric_name, metric_value)

        print("Logging model to MLflow...")
        mlflow.sklearn.log_model(best_model, "model")

        tmp_dir = "tmp_artifacts"
        os.makedirs(tmp_dir, exist_ok=True)

        # a. estimator.html
        estimator_html = sklearn.utils.estimator_html_repr(best_model)
        estimator_path = os.path.join(tmp_dir, "estimator.html")
        with open(estimator_path, "w", encoding="utf-8") as f:
            f.write(estimator_html)
        mlflow.log_artifact(estimator_path)
        print("Logged estimator.html")

        # b. metric_info.json
        metric_path = os.path.join(tmp_dir, "metric_info.json")
        with open(metric_path, "w") as f:
            json.dump(metrics, f, indent=4)
        mlflow.log_artifact(metric_path)
        print("Logged metric_info.json")

        # c. training_confusion_matrix.png
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=["Lancar", "Gagal Bayar"]
        )
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(cmap=plt.cm.Blues, ax=ax)
        plt.title("Confusion Matrix - Credit Scoring Test Set")
        plt.tight_layout()
        cm_path = os.path.join(tmp_dir, "training_confusion_matrix.png")
        plt.savefig(cm_path)
        plt.close(fig)
        mlflow.log_artifact(cm_path)
        print("Logged training_confusion_matrix.png")

        # d. feature_importance.png
        importances = best_model.feature_importances_
        feature_names = X.columns
        indices = np.argsort(importances)[::-1]

        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.barplot(x=importances[indices], y=feature_names[indices], color="steelblue", ax=ax2)
        ax2.set_title("Feature Importances - Random Forest Model")
        ax2.set_xlabel("Importance Score")
        ax2.set_ylabel("Features")
        plt.tight_layout()
        fi_path = os.path.join(tmp_dir, "feature_importance.png")
        fig2.savefig(fi_path)
        plt.close(fig2)
        mlflow.log_artifact(fi_path)
        print("Logged feature_importance.png (Additional Artifact 1)")

        # e. data_schema.json
        schema = {
            "features": [{"name": col, "type": str(df[col].dtype)} for col in X.columns],
            "target": {"name": "default", "type": str(df["default"].dtype)},
            "n_samples": len(df)
        }
        schema_path = os.path.join(tmp_dir, "data_schema.json")
        with open(schema_path, "w") as f:
            json.dump(schema, f, indent=4)
        mlflow.log_artifact(schema_path)
        print("Logged data_schema.json (Additional Artifact 2)")

        # Save model pickle
        os.makedirs("app", exist_ok=True)
        with open("app/model.pkl", "wb") as f:
            pickle.dump(best_model, f)
        print("Saved model pickle to app/model.pkl")

        # Save run ID
        with open("run_id.txt", "w") as f:
            f.write(run.info.run_id)

    print("Modelling and tuning completed successfully!")


if __name__ == "__main__":
    main()
