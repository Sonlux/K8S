import os
import pandas as pd
import numpy as np
import logging
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, f1_score
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Input, Concatenate, Attention, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.optimizers import Adam
import joblib
import tensorflow as tf

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
    parser = argparse.ArgumentParser(description='Train improved LSTM model for Kubernetes anomaly detection')
    parser.add_argument('--data_path', type=str, default='dataSynthetic.csv', help='Path to the input CSV data')
    parser.add_argument('--output_dir', type=str, default='model_artifacts', help='Directory to save model artifacts')
    parser.add_argument('--sequence_length', type=int, default=10, help='Sequence length for LSTM')
    parser.add_argument('--batch_size', type=int, default=32, help='Training batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--threshold_percentile', type=float, default=95, help='Percentile for anomaly threshold')
    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of data for testing')
    parser.add_argument('--use_attention', action='store_true', help='Use attention mechanism')
    parser.add_argument('--use_bidirectional', action='store_true', help='Use bidirectional LSTM layers')
    parser.add_argument('--use_autoencoder', action='store_true', help='Use autoencoder architecture')
    parser.add_argument('--use_robust_scaler', action='store_true', help='Use RobustScaler instead of MinMaxScaler')
    
    return parser.parse_args()

def prepare_output_directory(output_dir):
    """Create output directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(output_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    logger.info(f"Created output directory: {run_dir}")
    return run_dir

def load_and_preprocess_data(data_path):
    """Load and preprocess the dataset with improved feature engineering."""
    logger.info(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)
    
    # Convert timestamp to datetime
    if 'Timestamp' in df.columns:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        df = df.sort_values('Timestamp')
    
    # Define base features
    base_features = [
        'CPU Usage (%)', 'Memory Usage (%)', 'Pod Restarts',
        'Memory Usage (MB)', 'Network Receive Bytes', 'Network Transmit Bytes',
        'FS Reads Total (MB)', 'FS Writes Total (MB)',
        'Network Receive Packets Dropped (p/s)', 'Network Transmit Packets Dropped (p/s)',
        'Ready Containers'
    ]
    
    # Add derived features
    logger.info("Performing feature engineering")
    
    # Rate of change features
    for feature in base_features:
        if feature in df.columns:
            df[f'{feature}_rate'] = df[feature].diff().fillna(0)
            df[f'{feature}_rate_change'] = df[f'{feature}_rate'].diff().fillna(0)
    
    # Rolling statistics
    for feature in base_features:
        if feature in df.columns:
            df[f'{feature}_rolling_mean_3'] = df[feature].rolling(window=3, min_periods=1).mean()
            df[f'{feature}_rolling_std_3'] = df[feature].rolling(window=3, min_periods=1).std().fillna(0)
    
    # Container readiness ratio
    if 'Ready Containers' in df.columns and 'Total Containers' in df.columns:
        df['Container_Readiness_Ratio'] = df['Ready Containers'] / df['Total Containers'].replace(0, 1)
    elif 'Ready Containers' in df.columns:
        df['Container_Readiness_Ratio'] = df['Ready Containers']
    
    # Resource utilization ratios
    if 'CPU Usage (%)' in df.columns and 'CPU Limits (%)' in df.columns:
        df['CPU_Utilization_Ratio'] = df['CPU Usage (%)'] / df['CPU Limits (%)'].replace(0, 1)
    
    if 'Memory Usage (MB)' in df.columns and 'Memory Limits (MB)' in df.columns:
        df['Memory_Utilization_Ratio'] = df['Memory Usage (MB)'] / df['Memory Limits (MB)'].replace(0, 1)
    
    # Network throughput
    if 'Network Receive Bytes' in df.columns and 'Network Transmit Bytes' in df.columns:
        df['Network_Total_Bytes'] = df['Network Receive Bytes'] + df['Network Transmit Bytes']
    
    # I/O operations
    if 'FS Reads Total (MB)' in df.columns and 'FS Writes Total (MB)' in df.columns:
        df['FS_Total_Operations'] = df['FS Reads Total (MB)'] + df['FS Writes Total (MB)']
    
    # Collect all features
    all_features = [col for col in df.columns if col not in ['Timestamp', 'Pod Name', 'Namespace', 'Node Name', 'Pod Status', 'Event Reason', 'anomaly']]
    logger.info(f"Total features after engineering: {len(all_features)}")
    
    # Handle missing or zero values
    df[all_features] = df[all_features].fillna(0)
    
    # Define anomaly target with refined conditions
    logger.info("Defining anomaly labels with improved criteria")
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
    df.loc[df['Pod Restarts'] > 5, 'anomaly'] = 1  # Excessive restarts
    
    # Log anomaly distribution
    anomaly_count = df['anomaly'].sum()
    total_records = len(df)
    logger.info(f"Anomaly distribution: {anomaly_count}/{total_records} ({anomaly_count/total_records:.2%})")
    
    return df, all_features

def scale_features(df, features, run_dir, use_robust_scaler=False):
    """Scale features using MinMaxScaler or RobustScaler."""
    logger.info(f"Scaling features using {'RobustScaler' if use_robust_scaler else 'MinMaxScaler'}")
    
    if use_robust_scaler:
        scaler = RobustScaler()
    else:
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
    """Plot training history."""
    plt.figure(figsize=(12, 8))
    
    # Plot loss
    plt.subplot(2, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracy
    plt.subplot(2, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot AUC
    plt.subplot(2, 2, 3)
    plt.plot(history.history['auc'], label='Training AUC')
    plt.plot(history.history['val_auc'], label='Validation AUC')
    plt.title('Model AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.legend()
    
    # Plot precision
    plt.subplot(2, 2, 4)
    plt.plot(history.history['precision'], label='Training Precision')
    plt.plot(history.history['val_precision'], label='Validation Precision')
    plt.title('Model Precision')
    plt.xlabel('Epoch')
    plt.ylabel('Precision')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(run_dir, 'training_history.png'))
    logger.info(f"Training history plot saved to {os.path.join(run_dir, 'training_history.png')}")

def find_optimal_threshold(y_true, y_pred, run_dir):
    """Find optimal threshold using ROC curve and precision-recall curve."""
    logger.info("Finding optimal threshold")
    
    # ROC curve
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    
    # Precision-Recall curve
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred)
    pr_auc = auc(recall, precision)
    logger.info(f"Precision-Recall AUC: {pr_auc:.4f}")
    
    # Find threshold that maximizes F1 score
    f1_scores = []
    for threshold in pr_thresholds:
        y_pred_binary = (y_pred >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred_binary)
        f1_scores.append(f1)
    
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = pr_thresholds[optimal_idx]
    logger.info(f"Optimal threshold from F1 score: {optimal_threshold:.4f}")
    
    # Also calculate percentile-based threshold
    percentile_threshold = np.percentile(y_pred, 95)
    logger.info(f"Percentile-based threshold (95.0%): {percentile_threshold:.4f}")
    
    # Use the optimal threshold
    threshold = optimal_threshold
    logger.info(f"Final anomaly threshold: {threshold:.4f}")
    
    # Plot ROC curve
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(run_dir, 'roc_curve.png'))
    logger.info(f"ROC curve saved to {os.path.join(run_dir, 'roc_curve.png')}")
    
    # Plot Precision-Recall curve
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'Precision-Recall curve (AUC = {pr_auc:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.savefig(os.path.join(run_dir, 'precision_recall_curve.png'))
    logger.info(f"Precision-Recall curve saved to {os.path.join(run_dir, 'precision_recall_curve.png')}")
    
    return threshold

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

def build_attention_model(input_shape, metrics):
    """Build LSTM model with attention mechanism."""
    logger.info("Building LSTM model with attention mechanism")
    
    # Input layer
    inputs = Input(shape=input_shape)
    
    # Bidirectional LSTM layers
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = Dropout(0.3)(x)
    
    # Attention mechanism
    attention = Attention()([x, x])
    x = Concatenate()([x, attention])
    
    # Second LSTM layer
    x = Bidirectional(LSTM(64))(x)
    x = Dropout(0.3)(x)
    
    # Dense layers with layer normalization
    x = Dense(64, activation='relu')(x)
    x = LayerNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(32, activation='relu')(x)
    x = LayerNormalization()(x)
    
    # Output layer
    outputs = Dense(1, activation='sigmoid')(x)
    
    # Create model
    model = Model(inputs=inputs, outputs=outputs)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    model.summary(print_fn=logger.info)
    return model

def build_bidirectional_model(input_shape, metrics):
    """Build bidirectional LSTM model."""
    logger.info("Building bidirectional LSTM model")
    
    model = Sequential()
    
    # First bidirectional LSTM layer
    model.add(Bidirectional(LSTM(128, return_sequences=True, input_shape=input_shape)))
    model.add(Dropout(0.3))
    
    # Second bidirectional LSTM layer
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.3))
    
    # Dense layers with layer normalization
    model.add(Dense(64, activation='relu'))
    model.add(LayerNormalization())
    model.add(Dropout(0.2))
    
    model.add(Dense(32, activation='relu'))
    model.add(LayerNormalization())
    
    # Output layer
    model.add(Dense(1, activation='sigmoid'))
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    model.summary(print_fn=logger.info)
    return model

def build_autoencoder_model(input_shape, metrics):
    """Build autoencoder LSTM model for anomaly detection."""
    logger.info("Building autoencoder LSTM model")
    
    # Encoder
    inputs = Input(shape=input_shape)
    encoded = Bidirectional(LSTM(64, return_sequences=True))(inputs)
    encoded = Bidirectional(LSTM(32))(encoded)
    
    # Decoder
    decoded = tf.keras.layers.RepeatVector(input_shape[0])(encoded)
    decoded = Bidirectional(LSTM(32, return_sequences=True))(decoded)
    decoded = Bidirectional(LSTM(64, return_sequences=True))(decoded)
    decoded = tf.keras.layers.TimeDistributed(Dense(input_shape[1]))(decoded)
    
    # Reconstruction error
    reconstruction_error = tf.keras.layers.Lambda(
        lambda x: tf.reduce_mean(tf.square(x[0] - x[1]), axis=[1, 2])
    )([inputs, decoded])
    
    # Anomaly score
    anomaly_score = Dense(1, activation='sigmoid')(reconstruction_error)
    
    # Create model
    model = Model(inputs=inputs, outputs=anomaly_score)
    
    # Compile model
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=metrics
    )
    
    model.summary(print_fn=logger.info)
    return model

def build_model(input_shape, metrics, use_attention=False, use_bidirectional=False, use_autoencoder=False):
    """Build the appropriate LSTM model architecture based on parameters."""
    if use_autoencoder:
        return build_autoencoder_model(input_shape, metrics)
    elif use_attention:
        return build_attention_model(input_shape, metrics)
    elif use_bidirectional:
        return build_bidirectional_model(input_shape, metrics)
    else:
        logger.info("Building standard LSTM model")
        
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(128, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.3))
        
        # Second LSTM layer
        model.add(LSTM(64))
        model.add(Dropout(0.3))
        
        # Dense layers with layer normalization
        model.add(Dense(64, activation='relu'))
        model.add(LayerNormalization())
        model.add(Dropout(0.2))
        
        model.add(Dense(32, activation='relu'))
        model.add(LayerNormalization())
        
        # Output layer
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=0.001),
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
    df_scaled, scaler = scale_features(df, features, run_dir, args.use_robust_scaler)
    
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
    model = build_model(
        (X_train.shape[1], X_train.shape[2]), 
        metrics,
        args.use_attention,
        args.use_bidirectional,
        args.use_autoencoder
    )
    
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
        f.write(f"Model Architecture:\n")
        f.write(f"- Attention: {args.use_attention}\n")
        f.write(f"- Bidirectional: {args.use_bidirectional}\n")
        f.write(f"- Autoencoder: {args.use_autoencoder}\n")
        f.write(f"- Robust Scaler: {args.use_robust_scaler}\n\n")
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