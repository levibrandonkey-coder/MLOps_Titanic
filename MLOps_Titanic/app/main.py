import mlflow
import pandas as pd
from fastapi import FastAPI
import uvicorn

# Import helpers from utils
from utils import select_model, engineer_features, align_features

# Configure MLflow to point to your SQLite database
mlflow.set_tracking_uri("sqlite:///mlflow.db")

# Fallback loading logic if interactive select_model is not preferred at auto-startup
try:
    # We can use the utility logic to align our model loading
    experiment = mlflow.get_experiment_by_name("titanic_experiment")
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], order_by=["metrics.accuracy DESC"])
    best_run_id = runs.iloc[0]["run_id"]
    best_model_name = runs.iloc[0]["params.model_type"]
    model_uri = f"runs:/{best_run_id}/model"
    model = mlflow.sklearn.load_model(model_uri)
    print(f"Loaded best model '{best_model_name}' from run: {best_run_id}")
except Exception as e:
    print(f"Error loading MLflow model: {e}. Ensure training step has run.")
    best_run_id = "unknown"
    best_model_name = "unknown"
    model = None

app = FastAPI(title=f"Titanic Survival Prediction API (Model: {best_model_name})")

@app.get("/")
def root():
    return {
        "status": "active", 
        "loaded_model": best_model_name, 
        "loaded_run_id": best_run_id
    }

@app.post("/predict")
def predict(data: dict):
    if model is None:
        return {"error": "Model is not loaded. Run the training cell first."}
    
    # Load input dictionary into a pandas DataFrame
    df = pd.DataFrame([data])
    
    # Use the utilities logic for dynamic feature engineering and schema alignment
    df = engineer_features(df, model)
    df = align_features(df, model)
    
    # Generate prediction (0 = No, 1 = Yes)
    prediction = int(model.predict(df)[0])
    survival_status = "Survived" if prediction == 1 else "Did not survive"
    
    return {
        "prediction": prediction,
        "status": survival_status
    }

if __name__ == "__main__":
    # Run the server locally on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
