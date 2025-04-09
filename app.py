from fastapi import FastAPI
import pandas as pd
import joblib

app = FastAPI()

# Load trained AI model
model = joblib.load("k8s_failure_model.pkl")

@app.get("/")
def home():
    return {"message": "Kubernetes Failure Prediction API Running"}

@app.get("/predict")
def predict():
    try:
        # Load the latest Prometheus data
        df = pd.read_csv("prometheus_data.csv")

        # Make predictions
        df["failure_prediction"] = model.predict(df[['CPU(m)', 'MEMORY(Mi)']])
        df["failure_prediction"] = df["failure_prediction"].apply(lambda x: "YES" if x == 1 else "NO")

        return df[["pod", "CPU(m)", "MEMORY(Mi)", "failure_prediction"]].to_dict(orient="records")

    except Exception as e:
        return {"error": str(e)}
