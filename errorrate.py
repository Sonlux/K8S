def calculate_error_metrics(y_true, y_pred):
    """Calculate comprehensive error metrics"""
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, roc_auc_score
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate error rates
    false_positive_rate = fp / (fp + tn)
    false_negative_rate = fn / (fn + tp)
    
    # Calculate precision, recall, F1
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary')
    
    # Calculate AUC
    auc = roc_auc_score(y_true, y_pred)
    
    # Calculate accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    # Mean error rate
    mean_error_rate = (false_positive_rate + false_negative_rate) / 2
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'false_positive_rate': false_positive_rate,
        'false_negative_rate': false_negative_rate,
        'mean_error_rate': mean_error_rate
    }
