import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pickle

# Load the dataset
df = pd.read_csv("student_performance_tutor.csv")

# Encode categorical columns manually
def encode_all(df):
    df = df.copy()
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    df["school_level"] = df["school_level"].map({"Elementary": 0, "High School": 1})
    df["parental_support"] = df["parental_support"].map({"Low": 0, "Medium": 1, "High": 2})
    df["student_level"] = df["student_level"].map({"Beginner": 0, "Intermediate": 1, "Advanced": 2})
    df["focus_level"] = df["focus_level"].map({"Low": 0, "Medium": 1, "High": 2})
    df["internet_access"] = df["internet_access"].map({"No": 0, "Yes": 1})
    df["language_proficiency"] = df["language_proficiency"].map({"Basic": 0, "Intermediate": 1, "Fluent": 2})
    df["curriculum_type"] = df["curriculum_type"].map({"CBSE": 0, "ICSE": 1, "IB": 2, "State Board": 3})
    return df

# Apply encoding
df = encode_all(df)

# Split features and targets
X = df.drop(columns=["performance_score", "recommended_material"])
y_score = df["performance_score"]
y_recommendation = df["recommended_material"]

# Feature list (all are numeric now)
numerical_features = X.columns.tolist()

# Pipeline for performance score prediction
pipeline_reg = Pipeline([
    ("scaler", StandardScaler()),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train and save performance model
print("Training performance prediction model...")
pipeline_reg.fit(X, y_score)
with open("AssessScore.pkl", "wb") as f:
    pickle.dump(pipeline_reg, f)
print("Saved performance model to AssessScore.pkl")

# Pipeline for recommendation prediction
pipeline_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train and save recommendation model
print("Training recommendation model...")
pipeline_clf.fit(X, y_recommendation)
with open("recomendation.pkl", "wb") as f:
    pickle.dump(pipeline_clf, f)
print("Saved recommendation model to recomendation.pkl")

print("\u2705 Model training complete!")
