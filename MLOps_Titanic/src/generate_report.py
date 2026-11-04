import webbrowser
from pathlib import Path
import pandas as pd
import numpy as np
from evidently import Report
from evidently.presets import DataDriftPreset

# Set up directory paths
DATA_DIR = Path('data')
REPORTS_DIR = Path('reports')
REPORTS_DIR.mkdir(exist_ok=True)

# 1. Load dataset (processed_titanic.csv is in the current working directory)
df = pd.read_csv("processed_titanic.csv")

# Drop target label column if present
target_cols = [col for col in ["Survived"] if col in df.columns]
X = df.drop(columns=target_cols) if target_cols else df.copy()

# 2. Split into Reference (Baseline) and Current (Production) sets
X_ref = X.sample(frac=0.5, random_state=42).copy()
X_cur = X.sample(frac=0.5, random_state=1).copy()

# 3. Hardcode feature drift on specific Titanic columns
if 'Age' in X_cur.columns:
    X_cur['Age'] = X_cur['Age'] + 15.0  # Simulate an older passenger demographic

if 'Fare' in X_cur.columns:
    X_cur['Fare'] = X_cur['Fare'] * 2.5  # Simulate higher ticket prices

if 'SibSp' in X_cur.columns:
    X_cur['SibSp'] = X_cur['SibSp'] + 2  # Simulate larger families traveling

# 4. Hardcode prediction drift (simulating model output distribution shift)
# Reference predictions: Normal distribution of survival (0 = No, 1 = Yes)
X_ref['prediction'] = np.random.choice([0, 1], size=len(X_ref), p=[0.6, 0.4])

# Current predictions: Shifted distribution predicting lower survival rate due to demographic changes
X_cur['prediction'] = np.random.choice([0, 1], size=len(X_cur), p=[0.85, 0.15])

# 5. Run Evidently Data Drift Report (evaluates features + prediction column)
report = Report(metrics=[DataDriftPreset()])
snapshot = report.run(reference_data=X_ref, current_data=X_cur)

# 6. Save and open HTML report
output_path = REPORTS_DIR / "titanic_monitoring_report.html"
snapshot.save_html(str(output_path))

print(f"Report generated successfully with hardcoded drift at {output_path.resolve()}")
webbrowser.open(output_path.resolve().as_uri())
