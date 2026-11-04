from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Use SQLite backend in project root
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("titanic_experiment")

# Load the processed Titanic dataset
DATA_DIR = Path('data')
df = pd.read_csv(DATA_DIR / 'processed_titanic.csv')

# Separate features and target
X = df.drop('Survived', axis=1)
y = df['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models to train and evaluate
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
}

# Define trusted types for skops tree serialization
SKOPS_TRUSTED_TYPES = ["sklearn.tree._tree.Tree"]

# Train, evaluate, and log each model in its own MLflow run
for name, model in models.items():
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    with mlflow.start_run(run_name=name) as run:
        mlflow.log_param("model_type", name)

        # Log metrics
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", roc_auc)

        if hasattr(model, "get_params"):
            mlflow.log_params(model.get_params())

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            skops_trusted_types=SKOPS_TRUSTED_TYPES
        )

        print(f"[{name}] Logged run {run.info.run_id}")
        print(f"   Accuracy: {acc:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}\n")
