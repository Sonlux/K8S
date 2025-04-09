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
