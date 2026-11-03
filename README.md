# MLOps_Titanic
Develop a predictive model that accurately classifies whether passengers survived the Titanic disaster based on available demographic and travel-related features. Identify the key factors influencing passenger survival, build a robust classification model, evaluate its accuracy, and demo deployment and monitoring using an end-to-end MLOps pipeline.

# **Unit End Project**
## **Predictive Modeling and Deployment**
### **Problem Statement**
Develop a predictive model that accurately classifies whether passengers survived the Titanic disaster based on available demographic and travel-related features. The goal is to identify the key factors influencing passenger survival, build a robust classification model, evaluate its accuracy, and demonstrate its deployment and monitoring using an end-to-end MLOps pipeline.

### **Instructions:**
*  **Tools:** Python, FastAPI, MLflow, DVC, Evidently AI
*  **Dataset:** Titanic dataset containing demographic and travel data
*  **Submission:** Submit all Python scripts, API code, monitoring reports, and a project report through the LMS

### **Task**
Use Python and associated tools to develop an MLOps lifecycle project that includes the following:
*  Data exploration and preprocessing
*  Classification model development
*  Model and data versioning
*  CI/CD pipeline creation (optional)
*  Deployment of the model via FastAPI
*  Continuous monitoring using Evidently AI
*  Documentation of the entire workflow and best practices

### **Data Description**
The Titanic dataset contains demographic and passenger information for 891 individuals aboard the Titanic. Each record represents a passenger and includes the following attributes:

*  **PassengerId:** Unique identifier for each passenger.
*  **Survived:** Survival status (0 = No, 1 = Yes).
*  **Pclass:** Passenger class (1 = First Class, 2 = Second Class, 3 = Third Class).
*  **Name:** Passenger’s name.
*  **Sex:** Passenger’s gender (male or female).
*  **Age:** Age of passenger (some entries are missing).
*  **SibSp:** Number of siblings/spouses aboard the Titanic.
*  **Parch:** Number of parents/children aboard the Titanic.
*  **Ticket:** Ticket number.
*  **Fare:** Ticket fare paid.
*  **Cabin:** Cabin number (many entries are missing).
*  **Embarked:** Port of embarkation (C = Cherbourg, Q = Queenstown, S = Southampton).

## **Quickstart Guide**
1. Prerequisite: Git Setup
Ensure Git is installed (brew install git on macOS, sudo apt install git on Linux, or from git-scm.com).

2. Clone & Navigate
Clone the repo and enter the project directory:

git clone https://github.com/vivekfullstack/Fullstack.git
cd Fullstack
