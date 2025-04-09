def enhance_features(df):
    # Add error-specific features
    df['is_pod_error'] = df['Pod Status'].apply(lambda x: 1 if x in ['CrashLoopBackOff', 'Error', 'Unknown'] else 0)
    df['is_oom_killing'] = df['Event Reason'].apply(lambda x: 1 if x == 'OOMKilling' else 0)
    df['is_node_not_ready'] = df['Node Name'].str.contains('NodeNotReady', na=False).astype(int)
    
    # Calculate rolling statistics for key metrics
    for window in [5, 10, 30]:
        for feature in ['CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts']:
            df[f'{feature}_rolling_mean_{window}'] = df.groupby('Pod Name')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{feature}_rolling_std_{window}'] = df.groupby('Pod Name')[feature].transform(
                lambda x: x.rolling(window=window, min_periods=1).std())
    
    # Add rate of change features
    for feature in ['Memory Usage (%)', 'CPU Usage (%)']:
        df[f'{feature}_rate'] = df.groupby('Pod Name')[feature].transform(
            lambda x: x.diff() / x.shift(1))
    
    # Add pod restart acceleration
    df['restart_acceleration'] = df.groupby('Pod Name')['Pod Restarts'].transform(
        lambda x: x.diff().diff())
    
    return df
def generate_early_warnings(df, model, scaler, features, sequence_length, warning_threshold=0.7):
    # Define thresholds
    thresholds = {
        'general': 0.5,
        'memory_pct': 80,
        'memory_mb': 500,
        'cpu': 90,
        'restarts': 5,
        'network_receive': 10000,
        'network_transmit': 10000
    }
    
    # Get predictions with thresholds
    predictions = predict_anomalies(model, df, scaler, sequence_length, features, thresholds)
    
    # Rest of the function
