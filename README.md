# Improved LSTM Model for Kubernetes Anomaly Detection

This repository contains an improved LSTM (Long Short-Term Memory) model for detecting anomalies in Kubernetes cluster metrics. The model has been enhanced with several advanced features to improve detection accuracy and robustness.

## Features

- **Multiple Model Architectures**:

  - Standard LSTM
  - Bidirectional LSTM
  - LSTM with Attention Mechanism
  - LSTM Autoencoder

- **Advanced Feature Engineering**:

  - Rate of change features
  - Rolling statistics
  - Resource utilization ratios
  - Container readiness metrics
  - Network and I/O operation metrics

- **Robust Data Preprocessing**:

  - Support for both MinMaxScaler and RobustScaler
  - Time series-aware data splitting
  - Class weight balancing for imbalanced data

- **Comprehensive Evaluation**:

  - ROC curve analysis
  - Precision-Recall curve
  - Confusion matrix visualization
  - Detailed performance metrics

- **Training Visualization**:
  - Loss curves
  - Accuracy curves
  - AUC curves
  - Precision curves

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

Train the model with default settings:

```
python improved_lstm_model.py --data_path dataSynthetic.csv
```

Train with specific architecture:

```
python improved_lstm_model.py --data_path dataSynthetic.csv --use_attention --use_bidirectional
```

### Command Line Arguments

- `--data_path`: Path to the input CSV data (default: 'dataSynthetic.csv')
- `--output_dir`: Directory to save model artifacts (default: 'model_artifacts')
- `--sequence_length`: Sequence length for LSTM (default: 10)
- `--batch_size`: Training batch size (default: 32)
- `--epochs`: Number of training epochs (default: 50)
- `--threshold_percentile`: Percentile for anomaly threshold (default: 95)
- `--test_size`: Proportion of data for testing (default: 0.2)
- `--use_attention`: Use attention mechanism
- `--use_bidirectional`: Use bidirectional LSTM layers
- `--use_autoencoder`: Use autoencoder architecture
- `--use_robust_scaler`: Use RobustScaler instead of MinMaxScaler

## Model Outputs

The training process generates several artifacts in the output directory:

- `lstm_anomaly_model.h5`: Trained model
- `scaler.pkl`: Feature scaler
- `anomaly_threshold.pkl`: Optimal anomaly threshold
- `training_history.png`: Training metrics visualization
- `roc_curve.png`: ROC curve
- `precision_recall_curve.png`: Precision-Recall curve
- `confusion_matrix.png`: Confusion matrix
- `summary_report.txt`: Detailed training summary

## Anomaly Detection Criteria

The model considers the following conditions as anomalies:

1. Pod Status: CrashLoopBackOff, Error, Unknown
2. Event Reason: OOMKilling
3. Node Status: NodeNotReady
4. Network Issues: Dropped packets
5. Container Issues: Less than expected ready containers
6. Resource Usage:
   - Sudden CPU spikes
   - Very high memory usage (>95%)
   - Low container readiness ratio (<50%)
   - Excessive pod restarts (>5)

## Performance Metrics

The model is evaluated using multiple metrics:

- Accuracy
- AUC (Area Under the ROC Curve)
- Precision
- Recall
- F1 Score

## License

This project is licensed under the MIT License - see the LICENSE file for details.
