import pandas as pd
import numpy as np

np.random.seed(42)

# 15 symptoms
SYMPTOMS = [
    'fever', 'cough', 'headache', 'vomiting', 'fatigue',
    'body_ache', 'sore_throat', 'runny_nose', 'chest_pain',
    'shortness_of_breath', 'skin_rash', 'loss_of_taste',
    'diarrhea', 'chills', 'muscle_weakness'
]

# Each pattern: [fever, cough, headache, vomiting, fatigue,
#                body_ache, sore_throat, runny_nose, chest_pain,
#                shortness_of_breath, skin_rash, loss_of_taste,
#                diarrhea, chills, muscle_weakness]
PATTERNS = {
    'Flu':             [1,1,1,0,1, 1,1,0,0,0, 0,0,0,1,1],
    'Cold':            [0,1,0,0,0, 0,1,1,0,0, 0,0,0,0,0],
    'Migraine':        [0,0,1,1,0, 0,0,0,0,0, 0,0,0,0,0],
    'Typhoid':         [1,0,0,1,1, 1,0,0,0,0, 0,0,0,1,1],
    'Dengue':          [1,0,1,1,1, 1,0,0,0,0, 1,0,0,1,1],
    'Malaria':         [1,0,0,0,1, 1,0,0,0,0, 0,0,0,1,0],
    'Food Poisoning':  [0,0,0,1,1, 0,0,0,1,0, 0,0,1,0,0],
    'Asthma':          [0,1,0,0,1, 0,0,0,1,1, 0,0,0,0,0],
    'COVID-19':        [1,1,0,0,1, 1,1,0,0,1, 0,1,0,0,1],
    'Pneumonia':       [1,1,0,0,1, 1,0,0,1,1, 0,0,0,1,0],
    'Chickenpox':      [1,0,1,0,1, 1,0,0,0,0, 1,0,0,0,0],
    'Jaundice':        [1,0,1,1,1, 0,0,0,0,0, 1,0,0,0,1],
    'Sinusitis':       [0,0,1,0,0, 0,1,1,0,0, 0,0,0,0,0],
    'Gastroenteritis': [0,0,0,1,1, 1,0,0,1,0, 0,0,1,0,0],
    'Bronchitis':      [0,1,0,0,1, 1,1,0,1,1, 0,0,0,0,0],
}

SAMPLES_PER_DISEASE = 200
NOISE_RATE = 0.10  # 10% chance a symptom flips

records = []
for disease, pattern in PATTERNS.items():
    for _ in range(SAMPLES_PER_DISEASE):
        row = []
        for val in pattern:
            if np.random.random() < NOISE_RATE:
                row.append(1 - val)
            else:
                row.append(val)
        row.append(disease)
        records.append(row)

df = pd.DataFrame(records, columns=SYMPTOMS + ['disease'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset.csv', index=False)

print(f"✅ Dataset created: {len(df)} rows, {len(PATTERNS)} diseases, {len(SYMPTOMS)} symptoms")
print(df['disease'].value_counts())
