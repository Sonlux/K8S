import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

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

# Create target variable (1 for anomaly, 0 for normal)
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
        X.append(data[i:i+seq_length, :-1])  # All columns except the last (anomaly)
        y.append(data[i+seq_length, -1])     # Only the anomaly column
    return np.array(X), np.array(y)

# Prepare data for LSTM
sequence_length = 10
df_values = df_scaled.values
X, y = create_sequences(df_values, sequence_length)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=1)
# Save the model
model.save('lstm_anomaly_model.h5')
print("Model saved as 'lstm_anomaly_model.h5'")
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention, Input, Concatenate
from tensorflow.keras.models import Model

def build_lstm_model(input_shape):
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    lstm1 = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    lstm1 = Dropout(0.3)(lstm1)
    
    lstm2 = Bidirectional(LSTM(32, return_sequences=False))(lstm1)
    lstm2 = Dropout(0.3)(lstm2)
    
    # Dense layers
    dense1 = Dense(32, activation='relu')(lstm2)
    dense1 = Dropout(0.2)(dense1)
    
    # Output layer
    output = Dense(1, activation='sigmoid')(dense1)
    
    # Create model
    model = Model(inputs=inputs, outputs=output)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC', 'Precision', 'Recall']
    )
    
    return model
