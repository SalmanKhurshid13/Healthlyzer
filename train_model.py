import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Load dataset
data = pd.read_csv("dataset.csv")

SYMPTOM_COLS = [
    'fever', 'cough', 'headache', 'vomiting', 'fatigue',
    'body_ache', 'sore_throat', 'runny_nose', 'chest_pain',
    'shortness_of_breath', 'skin_rash', 'loss_of_taste',
    'diarrhea', 'chills', 'muscle_weakness'
]

X = data[SYMPTOM_COLS]
y = data['disease']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train — Random Forest with tuned params
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

# Cross-validation
cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')

print("=" * 55)
print("  HEALTHLYZER — MODEL TRAINING REPORT")
print("=" * 55)
print(f"  Training samples : {len(X_train)}")
print(f"  Test samples     : {len(X_test)}")
print(f"  Features         : {len(SYMPTOM_COLS)} symptoms")
print(f"  Classes          : {y.nunique()} diseases")
print(f"  Test Accuracy    : {acc:.2f}%")
print(f"  CV Accuracy (5)  : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
print("=" * 55)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
fi = pd.Series(model.feature_importances_, index=SYMPTOM_COLS).sort_values(ascending=False)
print("Top Feature Importances:")
for feat, imp in fi.items():
    bar = '█' * int(imp * 40)
    print(f"  {feat:<22} {bar} {imp:.3f}")

# Save
joblib.dump(model, "model.pkl")
print("\n✅ Model saved as model.pkl")
print(f"✅ Symptom columns: {SYMPTOM_COLS}")
