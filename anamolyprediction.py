def predict_anomalies(model, new_data, scaler, sequence_length, features, thresholds):
    """
    Predict anomalies with confidence scores and anomaly types
    
    Args:
        model: Trained LSTM model
        new_data: DataFrame with new metrics
        scaler: Fitted MinMaxScaler
        sequence_length: Length of sequences used for training
        features: List of feature names
        thresholds: Dictionary of thresholds for different anomaly types
        
    Returns:
        DataFrame with anomaly predictions and types
    """
    # Scale the features
    scaled_data = scaler.transform(new_data[features])
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length + 1):
        sequences.append(scaled_data[i:i+sequence_length])
    
    # Make predictions
    if sequences:
        predictions = model.predict(np.array(sequences))
        
        # Add predictions to the original data
        result = new_data.iloc[sequence_length-1:].copy()
        result['anomaly_probability'] = predictions
        result['predicted_anomaly'] = (predictions > thresholds['general']).astype(int)
        
        # Determine anomaly types
        result['anomaly_type'] = 'Normal'
        
        # Memory-related anomalies
        memory_mask = (
            (new_data['Memory Usage (%)'] > thresholds['memory_pct']) | 
            (new_data['Memory Usage (MB)'] > thresholds['memory_mb'])
        )
        result.loc[result.index.isin(new_data[memory_mask].index) & (result['predicted_anomaly'] == 1), 
                   'anomaly_type'] = 'Memory Exhaustion'
        
        # CPU-related anomalies
        cpu_mask = new_data['CPU Usage (%)'] > thresholds['cpu']
        result.loc[result.index.isin(new_data[cpu_mask].index) & (result['predicted_anomaly'] == 1), 
                   'anomaly_type'] = 'CPU Exhaustion'
        
        # Pod restart anomalies
        restart_mask = new_data['Pod Restarts'] > thresholds['restarts']
        result.loc[result.index.isin(new_data[restart_mask].index) & (result['predicted_anomaly'] == 1), 
                   'anomaly_type'] = 'Pod Stability Issue'
        
        # Network anomalies
        network_mask = (
            (new_data['Network Receive Bytes'] > thresholds['network_receive']) | 
            (new_data['Network Transmit Bytes'] > thresholds['network_transmit'])
        )
        result.loc[result.index.isin(new_data[network_mask].index) & (result['predicted_anomaly'] == 1), 
                   'anomaly_type'] = 'Network Issue'
        
        return result
    else:
        return pd.DataFrame()
