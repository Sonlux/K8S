import tensorflow as tf

def weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0):
    # Calculate binary crossentropy
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    
    # Apply weights
    weights = y_true * pos_weight + (1 - y_true)
    weighted_bce = bce * weights
    
    return tf.reduce_mean(weighted_bce)

# Use in model compilation
model.compile(
    optimizer='adam',
    loss=lambda y_true, y_pred: weighted_binary_crossentropy(y_true, y_pred, pos_weight=10.0),
    metrics=['accuracy', 'AUC', 'Precision', 'Recall']
)
