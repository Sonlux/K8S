import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("k8s_anomaly_detection")

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train LSTM model for Kubernetes anomaly detection')
    parser.add_argument('--data_path', type=str, default='dataSynthetic.csv', help='Path to the input CSV data')
    parser.add_argument('--output_dir', type=str, default='model_artifacts', help='Directory to save model artifacts')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length for LSTM')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--threshold_percentile', type=float, default=95, help='Percentile for anomaly threshold')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    
    return parser.parse_args()

def prepare_output_directory(output_dir):
    """Create output directory if it doesn't exist."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        logger.info(f"Created output directory: {output_dir}")
    
    # Create a timestamped subdirectory for this run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir)
    logger.info(f"Created run directory: {run_dir}")
    
    return run_dir

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset."""
    logger.info(f"Loading data from {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    
    # Convert timestamp and sort
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values('Timestamp')
    
    # Define features present in the dataset
    features = [
        'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts',
        'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
        'FS Reads Total (MB)', 'FS Writes Total (MB)',
        'Network Receive Packets Dropped (p/s)', 'Network Transmit Packets Dropped (p/s)',
        'Ready Containers'
    ]
    
    # Feature engineering: add derived features
    logger.info("Adding engineered features")
    
    # Handle infinite values and large numbers first
    for feature in features:
        # Replace infinite values with NaN
        df[feature] = df[feature].replace([np.inf, -np.inf], np.nan)
        # Cap large values at 99th percentile
        percentile_99 = df[feature].quantile(0.99)
        df[feature] = df[feature].clip(upper=percentile_99)
    
    # Calculate rates of change for key metrics (using pct_change and filling NaN with 0)
    for feature in ['CPU Usage (%)', 'Memory Usage (%)', 'Memory Usage (MB)']:
        df[f'{feature}_rate'] = df[feature].pct_change().fillna(0)
        # Cap rate of change at reasonable limits
        df[f'{feature}_rate'] = df[f'{feature}_rate'].clip(-1, 1)
    
    # Add rolling statistics (mean and std) with 5-period window
    for feature in ['CPU Usage (%)', 'Memory Usage (%)']:
        df[f'{feature}_rolling_mean'] = df[feature].rolling(window=5, min_periods=1).mean()
        df[f'{feature}_rolling_std'] = df[feature].rolling(window=5, min_periods=1).std().fillna(0)
    
    # Calculate container readiness ratio
    df['Container_Readiness_Ratio'] = df['Ready Containers'] / df['Total Containers'].replace(0, 1)
    df['Container_Readiness_Ratio'] = df['Container_Readiness_Ratio'].clip(0, 1)
    
    # Update features list with new engineered features
    engineered_features = [
        'CPU Usage (%)_rate', 'Memory Usage (%)_rate', 'Memory Usage (MB)_rate',
        'CPU Usage (%)_rolling_mean', 'CPU Usage (%)_rolling_std',
        'Memory Usage (%)_rolling_mean', 'Memory Usage (%)_rolling_std',
        'Container_Readiness_Ratio'
    ]
    
    all_features = features + engineered_features
    
    # Handle missing values
    df[all_features] = df[all_features].fillna(0)
    
    # Define anomaly target with refined conditions
    logger.info("Defining anomaly labels")
    df['anomaly'] = 0
    
    # Core anomaly conditions
    df.loc[df['Pod Status'].isin(['CrashLoopBackOff', 'Error', 'Unknown']), 'anomaly'] = 1
    df.loc[df['Event Reason'] == 'OOMKilling', 'anomaly'] = 1
    df.loc[df['Node Name'].str.contains('NodeNotReady', na=False), 'anomaly'] = 1
    df.loc[df['Network Receive Packets Dropped (p/s)'] > 0, 'anomaly'] = 1
    df.loc[df['Ready Containers'] < df['Total Containers'], 'anomaly'] = 1
    
    # Additional anomaly conditions based on engineered features
    df.loc[df['CPU Usage (%)_rate'] > 0.5, 'anomaly'] = 1  # Sudden CPU spike
    df.loc[df['Memory Usage (%)'] > 95, 'anomaly'] = 1     # Very high memory usage
    df.loc[df['Container_Readiness_Ratio'] < 0.5, 'anomaly'] = 1  # Less than half containers ready
    
    # Log anomaly distribution
    anomaly_count = df['anomaly'].sum()
    total_records = len(df)
    logger.info(f"Anomaly distribution: {anomaly_count}/{total_records} ({anomaly_count/total_records:.2%})")
    
    return df, all_features

def scale_features(df, features, run_dir):
    """Scale features using MinMaxScaler."""
    logger.info("Scaling features")
    
    scaler = MinMaxScaler()
    df_scaled = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)
    df_scaled['anomaly'] = df['anomaly']
    
    # Save the scaler
    scaler_path = os.path.join(run_dir, 'scaler.pkl')
    joblib.dump(scaler, scaler_path)
    logger.info(f"Scaler saved to {scaler_path}")
    
    return df_scaled, scaler

def create_sequences(data, seq_length):
    """Create sequences for LSTM model."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length, :-1])  # All features except anomaly
        y.append(data[i+seq_length, -1])     # Anomaly label
    return np.array(X), np.array(y)

def plot_training_history(history, run_dir):
    """Plot and save training history metrics."""
    # Plot training & validation accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    history_plot_path = os.path.join(run_dir, 'training_history.png')
    plt.savefig(history_plot_path)
    logger.info(f"Training history plot saved to {history_plot_path}")

def find_optimal_threshold(y_true, y_pred, run_dir):
    """Find optimal threshold using precision-recall curve."""
    logger.info("Finding optimal threshold using precision-recall curve")
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for thresh in thresholds:
        y_pred_binary = (y_pred >= thresh).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        f1_scores.append(f1)
    
    # Find threshold with highest F1 score
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    best_f1 = f1_scores[best_f1_idx]
    logger.info(f"Optimal threshold: {best_threshold:.4f} (F1={best_f1:.4f})")
    
    # Also calculate percentile threshold for comparison
    percentile_thresh = np.percentile(y_pred, 95)
    logger.info(f"95th percentile threshold: {percentile_thresh:.4f}")
    
    # Plot precision-recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, marker='.', label='Precision-Recall curve')
    plt.scatter(recall[best_f1_idx], precision[best_f1_idx], color='red', s=100, 
                label=f'Best threshold: {best_threshold:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve with Optimal Threshold')
    plt.legend()
    
    pr_curve_path = os.path.join(run_dir, 'precision_recall_curve.png')
    plt.savefig(pr_curve_path)
    
    return best_threshold

def evaluate_model(model, X_test, y_test, threshold, run_dir):
    """Evaluate model performance using various metrics."""
    logger.info("Evaluating model performance")
    
    # Get predictions
    y_pred_proba = model.predict(X_test)
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Log results
    logger.info(f"Confusion Matrix:\n{conf_matrix}")
    logger.info(f"Classification Report:\n{pd.DataFrame(class_report).transpose()}")
    
    # Save detailed evaluation results
    eval_results = {
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report
    }
    
    eval_path = os.path.join(run_dir, 'evaluation_results.joblib')
    joblib.dump(eval_results, eval_path)
    logger.info(f"Evaluation results saved to {eval_path}")
    
    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    classes = ['Normal', 'Anomaly']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)
    
    # Add text annotations
    thresh = conf_matrix.max() / 2
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            plt.text(j, i, format(conf_matrix[i, j], 'd'),
                    horizontalalignment="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    plt.savefig(cm_path)
    logger.info(f"Confusion matrix plot saved to {cm_path}")
    
def build_model(input_shape, metrics):
    """Build the LSTM model architecture."""
    logger.info("Building LSTM model")
    
    model = Sequential()
    
    # First LSTM layer with return sequences for stacking
    model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.3))
    
    # Second LSTM layer
    model.add(LSTM(64))
    model.add(Dropout(0.3))
    
    # Dense layers
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation='relu'))
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model with multiple metrics
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    model.summary(print_fn=logger.info)
    return model

def main():
    # Parse arguments
    args = parse_arguments()
    
    # Prepare output directory
    run_dir = prepare_output_directory(args.output_dir)
    
    # Save all parameters
    params_path = os.path.join(run_dir, 'parameters.joblib')
    joblib.dump(vars(args), params_path)
    logger.info(f"Parameters saved to {params_path}")
    
    # Load and preprocess data
    df, features = load_and_preprocess_data(args.data_path)
    
    # Scale features
    df_scaled, scaler = scale_features(df, features, run_dir)
    
    # Create sequences
    logger.info(f"Creating sequences with length {args.sequence_length}")
    df_values = df_scaled.values
    X, y = create_sequences(df_values, args.sequence_length)
    
    # Split into train and test sets using time series split
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
    
    logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    
    # Calculate class weights for imbalanced data
    class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    logger.info(f"Class weights: {class_weight_dict}")
    
    # Define metrics
    metrics = ['accuracy', AUC(), Precision(), Recall()]
    
    # Build the LSTM model
    model = build_model((X_train.shape[1], X_train.shape[2]), metrics)
    
    # Define callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ModelCheckpoint(os.path.join(run_dir, 'best_model.h5'), save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    ]
    
    # Train model
    logger.info(f"Training model for {args.epochs} epochs with batch size {args.batch_size}")
    history = model.fit(
        X_train, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_split=0.2,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=2
    )
    
    # Plot training history
    plot_training_history(history, run_dir)
    
    # Evaluate final model
    logger.info("Evaluating model on test set")
    loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test)
    logger.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test AUC: {auc:.4f}, Test Precision: {precision:.4f}, Test Recall: {recall:.4f}")
    
    # Predict anomaly scores on test set
    y_pred = model.predict(X_test).flatten()
    
    # Find optimal threshold
    threshold = find_optimal_threshold(y_test, y_pred, run_dir)
    
    # Evaluate model with optimal threshold
    evaluate_model(model, X_test, y_test, threshold, run_dir)
    
    # Save the model and threshold
    model_path = os.path.join(run_dir, 'lstm_anomaly_model.h5')
    model.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    threshold_path = os.path.join(run_dir, 'anomaly_threshold.pkl')
    joblib.dump(threshold, threshold_path)
    logger.info(f"Threshold saved to {threshold_path}")
    
    # Generate a summary report file
    with open(os.path.join(run_dir, 'summary_report.txt'), 'w') as f:
        f.write(f"K8s Anomaly Detection Model - Training Summary\n")
        f.write(f"==========================================\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Dataset: {args.data_path}\n")
        f.write(f"Total records: {len(df)}\n")
        f.write(f"Anomaly rate: {df['anomaly'].mean():.2%}\n\n")
        f.write(f"Model Performance:\n")
        f.write(f"- Accuracy: {accuracy:.4f}\n")
        f.write(f"- AUC: {auc:.4f}\n")
        f.write(f"- Precision: {precision:.4f}\n")
        f.write(f"- Recall: {recall:.4f}\n")
        f.write(f"- F1 Score: {2 * (precision * recall) / (precision + recall + 1e-10):.4f}\n\n")
        f.write(f"Optimal threshold: {threshold:.4f}\n")
        f.write(f"Training completed successfully.\n")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()