import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load and preprocess data
df = pd.read_csv('dataSynthetic.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.sort_values('Timestamp')

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

# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])
        y.append(data[i+seq_length, -1])
    return np.array(X), np.array(y)

sequence_length = 10
X, y = create_sequences(df_scaled.values, sequence_length)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
