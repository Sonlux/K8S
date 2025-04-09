import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model

# Load the data
df = pd.read_csv('dataSynthetic.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

# Define features for anomaly detection
features = [
    'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts', 
    'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
    'FS Reads Total (MB)', 'FS Writes Total (MB)'
]

# Create target variable
df['anomaly'] = 0
df.loc[df['Pod Status'] == 'CrashLoopBackOff', 'anomaly'] = 1
df.loc[df['Pod Status'] == 'Error', 'anomaly'] = 1
df.loc[df['Event Reason'] == 'OOMKilling', 'anomaly'] = 1
df.loc[df['Pod Status'] == 'Unknown', 'anomaly'] = 1
df.loc[df['Node Name'].str.contains('NodeNotReady', na=False), 'anomaly'] = 1

# Scale features
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
df_scaled['anomaly'] = df['anomaly']

# Create sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

# Prepare data for LSTM
sequence_length = 10
df_values = df_scaled.values
X, y = create_sequences(df_values, sequence_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the trained model
model = load_model('lstm_anomaly_model.h5')

# Now you can proceed with model evaluation
