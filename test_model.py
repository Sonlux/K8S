#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test script for the improved LSTM model for Kubernetes anomaly detection.
This script demonstrates how to load a trained model and use it for prediction.
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import seaborn as sns

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Test the improved LSTM model')
    parser.add_argument('--model_path', type=str, default='model_artifacts/lstm_anomaly_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--scaler_path', type=str, default='model_artifacts/scaler.pkl',
                        help='Path to the saved scaler')
    parser.add_argument('--threshold_path', type=str, default='model_artifacts/anomaly_threshold.pkl',
                        help='Path to the saved anomaly threshold')
    parser.add_argument('--test_data', type=str, default='dataSynthetic.csv',
                        help='Path to test data CSV file')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for LSTM input')
    parser.add_argument('--output_dir', type=str, default='test_results',
                        help='Directory to save test results')
    return parser.parse_args()

def prepare_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def load_and_preprocess_data(data_path, scaler, sequence_length):
    """Load and preprocess test data."""
    # Load data
    df = pd.read_csv(data_path)
    
    # Extract features (same as in training)
    features = [
        'cpu_usage', 'memory_usage', 'network_rx', 'network_tx',
        'disk_read', 'disk_write', 'container_ready_ratio',
        'pod_restart_count', 'node_ready_ratio', 'network_dropped_packets'
    ]
    
    # Scale features
    X = scaler.transform(df[features])
    
    # Create sequences
    X_sequences = []
    for i in range(len(X) - sequence_length + 1):
        X_sequences.append(X[i:(i + sequence_length)])
    
    return np.array(X_sequences)

def plot_roc_curve(y_true, y_pred, output_dir):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'test_roc_curve.png'))
    plt.close()
    
    return roc_auc

def plot_precision_recall_curve(y_true, y_pred, output_dir):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, 'test_precision_recall_curve.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred_binary, output_dir):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred_binary)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(os.path.join(output_dir, 'test_confusion_matrix.png'))
    plt.close()
    
    return cm

def evaluate_predictions(y_true, y_pred, y_pred_binary, output_dir):
    """Evaluate model predictions and save results."""
    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    accuracy = accuracy_score(y_true, y_pred_binary)
    precision = precision_score(y_true, y_pred_binary)
    recall = recall_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)
    
    # Plot ROC curve and get AUC
    auc_score = plot_roc_curve(y_true, y_pred, output_dir)
    
    # Plot Precision-Recall curve
    plot_precision_recall_curve(y_true, y_pred, output_dir)
    
    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred_binary, output_dir)
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'test_metrics.txt'), 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"AUC: {auc_score:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")
        f.write("\nConfusion Matrix:\n")
        f.write(f"True Negatives: {cm[0,0]}\n")
        f.write(f"False Positives: {cm[0,1]}\n")
        f.write(f"False Negatives: {cm[1,0]}\n")
        f.write(f"True Positives: {cm[1,1]}\n")
    
    return {
        'accuracy': accuracy,
        'auc': auc_score,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def main():
    """Main function to test the model."""
    # Parse arguments
    args = parse_arguments()
    
    # Prepare output directory
    output_dir = prepare_output_directory(args.output_dir)
    
    # Load model and artifacts
    print("Loading model and artifacts...")
    model = load_model(args.model_path)
    scaler = joblib.load(args.scaler_path)
    threshold = joblib.load(args.threshold_path)
    
    # Load and preprocess test data
    print("Loading and preprocessing test data...")
    X_test = load_and_preprocess_data(args.test_data, scaler, args.sequence_length)
    
    # Make predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    
    # Apply threshold to get binary predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # For evaluation, we need true labels
    # In a real scenario, you would have these from your test data
    # For this example, we'll generate random labels for demonstration
    y_true = np.random.randint(0, 2, size=len(y_pred))
    
    # Evaluate predictions
    print("Evaluating predictions...")
    metrics = evaluate_predictions(y_true, y_pred, y_pred_binary, output_dir)
    
    # Print results
    print("\nTest Results:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"\nResults saved to {output_dir}/")

if __name__ == "__main__":
    main() 