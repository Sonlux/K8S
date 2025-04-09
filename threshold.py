def optimize_thresholds(model, X_val, y_val, initial_threshold=0.5, step=0.01):
    """Find optimal threshold to balance precision and recall"""
    from sklearn.metrics import precision_recall_curve, f1_score
    
    # Get predictions
    y_pred_proba = model.predict(X_val)
    
    # Calculate precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_val, y_pred_proba)
    
    # Calculate F1 score for each threshold
    f1_scores = []
    for i in range(len(precision)):
        if i < len(thresholds):
            threshold = thresholds[i]
        else:
            threshold = 0.0
        y_pred = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred)
        f1_scores.append((threshold, f1))
    
    # Find threshold with highest F1 score
    optimal_threshold = max(f1_scores, key=lambda x: x[1])[0]
    
    return optimal_threshold
