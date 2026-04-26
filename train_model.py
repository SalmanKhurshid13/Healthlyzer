import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib

# Load dataset
data = pd.read_csv("dataset.csv")

X = data.drop("disease", axis=1)
y = data["disease"]

# Train model
model = LogisticRegression(max_iter=200)
model.fit(X, y)

# Save trained model
joblib.dump(model, "model.pkl")

print("Model trained successfully!")
print(list(X.columns))