# Titanic Survivor MLOps

A predictive model that accurately classifies whether passengers survived the Titanic disaster based on available demographic and travel-related features.

---

## 📌 Features

* **Data Preprocessing**: Cleaning, feature engineering, and category encoding.
* **Experiment Tracking**: Training multiple models (Random Forest, Logistic Regression) with MLflow logging parameters, metrics, and serialized model artifacts.
* **Model Registry & UI**: SQLite-backed MLflow tracking server and UI dashboard.
* **Model Serving**: Web application/API serving real-time predictions.

---

## 📍 Prerequisite: Set Project Directory


Verify your current working directory:

```bash
pwd
# Output should end with: .../MLOps_Titanic
```

Before executing any commands, navigate to the project directory:

```bash
cd MLOps_Titanic
```

---

## 📁 Project Structure

```text
Fullstack/
└── mlops_demo/
    ├── app/                    # API / Web server for inference
    ├── data/                   # Raw and processed datasets
    ├── mlruns/                 # MLflow artifact storage directory
    ├── reports/                # Evaluation & analysis reports
    ├── samples/                # Sample input data for testing
    ├── src/
    │   ├── data_preprocessing.py   # Dataset processing & feature creation
    │   └── model_tracking_mlflow.py # Model training & MLflow logging script
    ├── venv/                   # Virtual environment
    ├── mlflow.db               # SQLite database for MLflow backend store
    ├── .gitignore              # Git ignore configuration
    ├── README.md               # Project documentation
    └── requirements.txt        # Python dependencies
```

---

## ⚙️ Setup Instructions

### 1. Create and Activate Virtual Environment

```bash
python -m venv venv
source venv/bin/activate
```

*On Windows (Git Bash / CMD):*

```bash
venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Pipeline Execution

### Step 1: Preprocess the Dataset

Clean raw data and encode categories as necessary.

```bash
python src/data_preprocessing.py
```

### Step 2: Train & Track Models with MLflow

Train classification models (e.g., `RandomForestClassifier`, `LogisticRegression`) and log metrics, parameters, and artifacts to SQLite.

```bash
python src/model_tracking_mlflow.py
```

### Step 3: Launch MLflow Tracking UI

Start the local MLflow dashboard to view experiment runs, compare model accuracy, and inspect saved artifacts.

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5001
```

Access the UI in your web browser at:  
👉 **`http://127.0.0.1:5001`**

### Step 4: Run the Prediction Application

In a separate terminal window (ensure you are inside `/Users/levibrandonkey-coder/MLOps_Titanic` with the `venv` activated), launch the application:

```bash
python app/main.py
```

### Step 5: Test Model Predictions (Interactive API Docs)

1. Open **`http://127.0.0.1:8000/docs`** in your web browser to access the Swagger UI.
2. Click on the **`POST /predict`** endpoint and click **Try it out**.
3. Copy the sample JSON payload from the `samples/` directory and paste them into the request body.
4. Click **Execute** to view the live inference output and confidence scores.

### Step 6: Automated Data & Prediction Drift Monitoring

To ensure the model performs reliably over time, we use **Evidently AI** to monitor incoming inference data for covariate shift (feature drift) and concept shift (prediction drift) against our baseline dataset.

Execute the report generator script to compare the reference (training) dataset against current production inputs:

```bash
python src/generate_report.py
