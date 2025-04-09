# earlywarning.py
import pandas as pd

# Import your anomaly prediction function
from anamolyprediction import predict_anomalies


def generate_early_warnings(df, model, scaler, features, sequence_length, warning_threshold=0.7):
    """
    Generate early warnings for potential issues
    
    Args:
        df: DataFrame with metrics
        model: Trained LSTM model
        scaler: Fitted MinMaxScaler
        features: List of feature names
        sequence_length: Length of sequences used for training
        warning_threshold: Probability threshold for generating warnings
        
    Returns:
        DataFrame with warnings
    """
    # Get predictions
    predictions = predict_anomalies(model, df, scaler, sequence_length, features)
    
    # Generate warnings for high probability anomalies
    warnings = predictions[predictions['anomaly_probability'] > warning_threshold].copy()
    
    # Categorize warnings
    warnings['warning_type'] = 'Unknown'
    
    # Memory-related warnings
    memory_cols = [col for col in features if 'Memory' in col]
    for pod in warnings['Pod Name'].unique():
        pod_data = df[df['Pod Name'] == pod]
        for col in memory_cols:
            if pod_data[col].iloc[-1] > 80:  # If memory usage > 80%
                warnings.loc[warnings['Pod Name'] == pod, 'warning_type'] = 'Memory Exhaustion Risk'
    
    # CPU-related warnings
    if 'CPU Usage (%)' in features:
        cpu_high = df['CPU Usage (%)'] > 90
        warnings.loc[warnings.index.isin(df[cpu_high].index), 'warning_type'] = 'CPU Exhaustion Risk'
    
    # Pod restart warnings
    if 'Pod Restarts' in features:
        high_restarts = df['Pod Restarts'] > 5
        warnings.loc[warnings.index.isin(df[high_restarts].index), 'warning_type'] = 'Pod Stability Issue'
    
    return warnings

# Test the early warning system
if __name__ == "__main__":
    # Import necessary components
    from tensorflow.keras.models import load_model
    from sklearn.preprocessing import MinMaxScaler
    import pandas as pd
    
    # Load your trained model (if saved) or use the one from lstmmodel.py
    try:
        model = load_model('lstm_anomaly_model.h5')
    except:
        print("Model not found. Please run lstmmodel.py first and save the model.")
        exit()
    
    # Load sample data
    df = pd.read_csv('dataSynthetic.csv')
    
    # Define features
    features = [
        'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts', 
        'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
        'FS Reads Total (MB)', 'FS Writes Total (MB)'
    ]
    
    # Create a scaler and fit it to the data
    scaler = MinMaxScaler()
    scaler.fit(df[features])
    
    # Use a sample of the data for testing
    test_sample = df.iloc[-100:].copy()
    
    # Generate warnings
    sequence_length = 10
    warnings = generate_early_warnings(df, model, scaler, features, sequence_length)
    
    # Print results
    if not warnings.empty:
        print(f"Warnings generated: {len(warnings)}")
        print("\nSample warnings:")
        print(warnings[['Pod Name', 'warning_type', 'anomaly_probability']].head(10))
    else:
        print("No warnings generated for the sample data.")
