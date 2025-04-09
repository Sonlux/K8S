from tensorflow.keras.layers import Bidirectional, RepeatVector, TimeDistributed

# Build an autoencoder LSTM for multivariate anomaly detection
def build_autoencoder(input_shape):
    model = Sequential()
    # Encoder
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True, input_shape=input_shape)))
    model.add(Bidirectional(LSTM(32, activation='relu')))
    # Decoder
    model.add(RepeatVector(input_shape[0]))
    model.add(Bidirectional(LSTM(32, activation='relu', return_sequences=True)))
    model.add(Bidirectional(LSTM(64, activation='relu', return_sequences=True)))
    model.add(TimeDistributed(Dense(input_shape[1])))
    
    model.compile(optimizer='adam', loss='mse')
    return model
