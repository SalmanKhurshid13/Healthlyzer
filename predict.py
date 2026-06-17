import joblib
import pandas as pd

model = joblib.load("model.pkl")

SYMPTOM_COLS = [
    'fever', 'cough', 'headache', 'vomiting', 'fatigue',
    'body_ache', 'sore_throat', 'runny_nose', 'chest_pain',
    'shortness_of_breath', 'skin_rash', 'loss_of_taste',
    'diarrhea', 'chills', 'muscle_weakness'
]

# Example: COVID-19 pattern — fever + cough + fatigue + loss_of_taste + muscle_weakness
sample = pd.DataFrame([[1, 1, 0, 0, 1,  1, 1, 0, 0, 1,  0, 1, 0, 0, 1]], columns=SYMPTOM_COLS)
print("Predicted Disease:", model.predict(sample)[0])

# Example: Dengue pattern — fever + headache + vomiting + fatigue + skin_rash + chills
sample2 = pd.DataFrame([[1, 0, 1, 1, 1,  1, 0, 0, 0, 0,  1, 0, 0, 1, 1]], columns=SYMPTOM_COLS)
print("Predicted Disease:", model.predict(sample2)[0])
