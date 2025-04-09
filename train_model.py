import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Simulated dataset
data = {
    "CPU(m)": [100, 200, 150, 90, 300],
    "MEMORY(Mi)": [200, 300, 250, 180, 400],
    "Failure_Label": [0, 1, 0, 0, 1],  # 1 = Failure, 0 = No Failure
}
df = pd.DataFrame(data)

# Split dataset
X = df[['CPU(m)', 'MEMORY(Mi)']]
y = df['Failure_Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train AI model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model as k8s_failure_model.pkl
joblib.dump(model, "k8s_failure_model.pkl")

print("✅ Model trained and saved as k8s_failure_model.pkl")
