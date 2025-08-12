import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pickle
import os

# Create folders if not exist
os.makedirs("model", exist_ok=True)
os.makedirs("data", exist_ok=True)

# Load dataset
df = pd.read_csv('smart_retail_forecaster/data/sales_data.csv')

# Feature engineering
df['date'] = pd.to_datetime(df['date'])
df['day'] = df['date'].dt.day
df['month'] = df['date'].dt.month
df['year'] = df['date'].dt.year

X = df[['day', 'month', 'year', 'is_promo']]
y = df['sales']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model
with open('smart_retail_forecaster\model\demand_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved successfully!")
