import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# Load dataset
df = pd.read_csv("FairTipRecommender/data/tips.csv")

# Features & target
X = df.drop("tip", axis=1)
y = df["tip"]

# Preprocessing
categorical_features = ["time_of_day", "gender"]
numeric_features = ["total_bill", "service_rating", "group_size"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(drop="first"), categorical_features),
        ("num", "passthrough", numeric_features)
    ]
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(n_estimators=100, random_state=42))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "FairTipRecommender/model/tip_model.pkl")
print("Model saved to FairTipRecommender/model/tip_model.pkl")