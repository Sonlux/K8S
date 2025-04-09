# app.py
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import time
import os

# Import your functions
from anamolyprediction import predict_anomalies
from earlywarning import generate_early_warnings

def main():
    """Main function to run the anomaly detection system"""
    print("Starting Kubernetes Anomaly Detection System...")
    
    # Load your trained model
    try:
        model = load_model('lstm_anomaly_model.h5')
        print("Model loaded successfully.")
    except:
        print("Error: Model not found. Please run lstmmodel.py first and save the model.")
        return
    
    # Define features
    features = [
        'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts', 
        'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
        'FS Reads Total (MB)', 'FS Writes Total (MB)'
    ]
    
    # Create and fit scaler
    print("Loading data and preparing scaler...")
    df = pd.read_csv('dataSynthetic.csv')
    scaler = MinMaxScaler()
    scaler.fit(df[features])
    
    # Define sequence length
    sequence_length = 10
    
    # In a real-world scenario, you would fetch metrics from Kubernetes
    # For this demo, we'll use the dataset to simulate real-time data
    print("Starting monitoring loop...")
    
    # Simulate monitoring by processing chunks of data
    chunk_size = 20  # Process 20 records at a time
    for i in range(0, len(df) - chunk_size, chunk_size):
        # Get a chunk of data
        chunk = df.iloc[i:i+chunk_size+sequence_length].copy()
        
        # Generate warnings
        warnings = generate_early_warnings(chunk, model, scaler, features, sequence_length)
        
        # Print warnings
        if not warnings.empty:
            print(f"\n[{pd.Timestamp.now()}] WARNINGS DETECTED:")
            for _, warning in warnings.iterrows():
                print(f"  - {warning['Pod Name']}: {warning['warning_type']} (Probability: {warning['anomaly_probability']:.2f})")
        else:
            print(f"\n[{pd.Timestamp.now()}] No anomalies detected in current window.")
        
        # Wait to simulate real-time monitoring
        time.sleep(1)
    
    print("\nMonitoring complete.")

if __name__ == "__main__":
    main()
