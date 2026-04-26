import joblib
import pandas as pd

model = joblib.load("model.pkl")

columns = ['fever', 'cough', 'headache', 'vomiting', 'fatigue']

sample_input = pd.DataFrame([[1, 0, 1, 0, 1]], columns=columns)

prediction = model.predict(sample_input)

print("Predicted Disease:", prediction[0])