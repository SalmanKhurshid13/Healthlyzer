import pandas as pd
import numpy as np
np.random.seed(42)

records = []

# More DISTINCT patterns - each disease has unique symptom combination
patterns = {
    'Flu':            [1, 1, 1, 0, 1],  # fever+cough+headache+fatigue
    'Cold':           [0, 1, 0, 0, 0],  # cough only dominant
    'Migraine':       [0, 0, 1, 1, 0],  # headache+vomiting only
    'Typhoid':        [1, 0, 0, 1, 1],  # fever+vomiting+fatigue
    'Dengue':         [1, 0, 1, 1, 1],  # fever+headache+vomiting+fatigue
    'Malaria':        [1, 0, 0, 0, 1],  # fever+fatigue only
    'Food Poisoning': [0, 0, 0, 1, 0],  # vomiting only dominant
    'Asthma':         [0, 1, 0, 0, 1],  # cough+fatigue
    'COVID-19':       [1, 1, 0, 0, 1],  # fever+cough+fatigue (no headache)
}

for disease, pattern in patterns.items():
    for _ in range(80):  # 80 per disease = 720 total
        row = []
        for val in pattern:
            # 90% chance keeps the pattern — more consistent data
            if np.random.random() < 0.90:
                row.append(val)
            else:
                row.append(1 - val)
        row.append(disease)
        records.append(row)

df = pd.DataFrame(records, columns=['fever','cough','headache','vomiting','fatigue','disease'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df.to_csv('dataset.csv', index=False)
print('Total rows:', len(df))
print(df['disease'].value_counts())