from pathlib import Path
import pandas as pd

print("Starting Wine Dataset Preprocessing Pipeline... \n")

DATA_DIR = Path('data')
raw_file = DATA_DIR / 'titanic.csv'
processed_file = DATA_DIR / 'processed_titanic.csv'

# 2. Basic Preprocessing & Feature Engineering
print("Extracting 'Title' from 'Name'...")
df['Title'] = df['Name'].str.extract(r' ([A-Za-z]+)\.', expand=False)
# Replace rare titles
rare_titles = ['Lady', 'Countess','Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona']
df['Title'] = df['Title'].replace(rare_titles, 'Rare')
df['Title'] = df['Title'].replace({'Mlle': 'Miss', 'Ms': 'Miss', 'Mme': 'Mrs'})

print("Creating 'FamilySize' feature...")
df['FamilySize'] = df['SibSp'] + df['Parch'] + 1

print("Filling missing 'Age' values with the median...")
df['Age'] = df['Age'].fillna(df['Age'].median())

print("Categorizing 'Age' into ['child', 'adult', 'senior']...")
bins = [0, 12, 60, 100]
labels = ["child", "adult", "senior"]
df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)

# 3. Drop Unnecessary Columns
print("Dropping 'Cabin', 'Name', 'Ticket', and 'PassengerId' columns...")
df.drop(columns=['Cabin', 'Name', 'Ticket', 'PassengerId'], inplace=True, errors='ignore')

# 4. Encode Categorical Variables
print("Encoding categorical variables...")
# Fill missing Embarked values with the mode before encoding
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# One-hot encode categorical features
df = pd.get_dummies(df, columns=['Sex', 'Embarked', 'age_group', 'Title'], drop_first=True)

# 5. Save Processed Data
print(f"Saving processed dataset to: {processed_file}...")
processed_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(processed_file, index=False)

print(f"\nPreprocessing completed successfully!")
print(f"   • Final dataset shape: {df.shape[0]} rows, {df.shape[1]} columns")
