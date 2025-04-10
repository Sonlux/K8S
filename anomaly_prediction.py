import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("anomaly_prediction")

model = load_model('D:\\K8\\lstm_anomaly_model.h5')
scaler = joblib.load('D:\\K8\\scaler.pkl')
threshold = joblib.load('D:\\K8\\anomaly_threshold.pkl')
logger.info(f"Loaded model, scaler, and threshold: {threshold:.4f}")

# Use 11 features to match trained model
features = [
    'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts',
    'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
    'FS Reads Total (MB)', 'FS Writes Total (MB)',
    'Network Receive Packets Dropped (p/s)', 'Network Transmit Packets Dropped (p/s)',
    'Ready Containers'
]

def predict_anomalies(data, sequence_length=10):
    missing_cols = [col for col in features if col not in data.columns]
    if missing_cols:
        logger.warning(f"Missing columns {missing_cols}, filling with zeros")
        for col in missing_cols:
            data[col] = 0
    data = data[features].copy()
    logger.debug(f"Raw metrics (last row): {data.iloc[-1].to_dict()}")

    if len(data) < sequence_length:
        padded_data = np.zeros((sequence_length - len(data), len(features)))
        scaled_data = scaler.transform(np.vstack((padded_data, data)))
    else:
        scaled_data = scaler.transform(data.tail(sequence_length))

    X = scaled_data.reshape(1, sequence_length, len(features))
    prediction = model.predict(X, verbose=0)
    anomaly_score = prediction[0][0]
    is_anomaly = anomaly_score > threshold
    logger.debug(f"Model prediction - anomaly_score: {anomaly_score:.4f}, threshold: {threshold:.4f}, is_anomaly: {is_anomaly}")

    anomaly_type = 'unknown'
    last_row = data.iloc[-1]
    if is_anomaly:
        if last_row['Pod Restarts'] > 5:
            anomaly_type = 'crash_loop'
        elif last_row['Memory Usage (MB)'] > 500:  # No Memory Limits Utilization (%)
            anomaly_type = 'oom_kill'
        elif last_row['CPU Usage (%)'] > 90:
            anomaly_type = 'resource_exhaustion'
        elif (last_row['Network Receive Packets Dropped (p/s)'] > 0 or 
              last_row['Network Transmit Packets Dropped (p/s)'] > 0 or 
              last_row['Network Transmit Bytes'] > 10000):
            anomaly_type = 'network_issue'
        elif last_row['Ready Containers'] < last_row.get('Total Containers', 1):
            anomaly_type = 'partial_failure'
        elif last_row['FS Reads Total (MB)'] > 10 or last_row['FS Writes Total (MB)'] > 10:
            anomaly_type = 'io_issue'

    result = pd.DataFrame({
        'predicted_anomaly': [1 if is_anomaly else 0],
        'anomaly_probability': [anomaly_score],
        'anomaly_type': [anomaly_type]
    })
    logger.debug(f"Prediction result: {result.iloc[0].to_dict()}")
    return result

if __name__ == "__main__":
    sample_data = {
        'CPU Usage (%)': [0.012098631],
        'Memory Usage (%)': [4.747099786],
        'Pod Restarts': [370],
        'Memory Usage (MB)': [18.47939985],
        'Network Receive Bytes': [0.014544437],
        'Network Transmit Bytes': [0.122316709],
        'FS Reads Total (MB)': [0.000586664],
        'FS Writes Total (MB)': [0.000836434],
        'Network Receive Packets Dropped (p/s)': [0],
        'Network Transmit Packets Dropped (p/s)': [0],
        'Ready Containers': [0]
    }
    df_sample = pd.DataFrame(sample_data)
    prediction_df = predict_anomalies(df_sample)
    logger.info("\nFinal Prediction:")
    logger.info(prediction_df.to_string())