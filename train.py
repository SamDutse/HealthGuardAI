import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# -------------------------
# 1. Generate Synthetic Data
# -------------------------

np.random.seed(42)
n = 500

data = pd.DataFrame({
    "age": np.random.randint(16, 45, n),
    "parity": np.random.randint(0, 6, n),
    "anc_visits": np.random.randint(0, 8, n),
    "malaria": np.random.randint(0, 2, n),
    "hemoglobin": np.random.uniform(7, 14, n),
    "nutrition_score": np.random.randint(1, 6, n),
    "distance_km": np.random.uniform(1, 30, n),
    "prev_complication": np.random.randint(0, 2, n),
    "education_level": np.random.randint(0, 3, n)
})

# -------------------------
# 2. Define Risk Logic
# -------------------------

risk_score = (
    (data["age"].apply(lambda x: 1 if x < 18 or x > 35 else 0)) * 1 +
    (data["anc_visits"] < 4) * 2 +
    (data["hemoglobin"] < 10) * 2 +
    (data["malaria"] == 1) * 1 +
    (data["prev_complication"] == 1) * 3 +
    (data["distance_km"] > 15) * 1
)

data["high_risk"] = (risk_score >= 4).astype(int)


# -------------------------
# 3. Split Data
# -------------------------

X = data.drop("high_risk", axis=1)
y = data["high_risk"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------------
# 4. Train Model
# -------------------------

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# -------------------------
# 5. Evaluate Model
# -------------------------

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

plt.figure()
sns.heatmap(cm, annot=True, fmt="d")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

joblib.dump(model, "healthguard_model.pkl")
print("Model saved as healthguard_model.pkl")