import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# =============================================================================
# DATA PREPROCESSING FOR NEURAL NETWORKS
# =============================================================================

def prepare_data_for_nn(X_train, X_test, y_train, y_test):
    """
    Prepare data for neural networks - scaling is important!
    """
    print("=== PREPARING DATA FOR NEURAL NETWORKS ===")
    
    # Scale features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Training samples: {X_train_scaled.shape[0]}")
    print(f"Test samples: {X_test_scaled.shape[0]}")
    print(f"Features: {X_train_scaled.shape[1]}")
    print(f"Class distribution - Train: {np.bincount(y_train)}")
    print(f"Class distribution - Test: {np.bincount(y_test)}")
    print()
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

# =============================================================================
# MODEL 1: SIMPLE FEEDFORWARD NETWORK
# =============================================================================

def create_simple_nn(input_dim, learning_rate=0.001):
    """
    Simple feedforward neural network - good starting point
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Binary classification
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# =============================================================================
# MODEL 2: DEEPER NETWORK WITH BATCH NORMALIZATION
# =============================================================================

def create_deep_nn(input_dim, learning_rate=0.001):
    """
    Deeper network with batch normalization for better training
    """
    model = models.Sequential([
        layers.Dense(256, input_shape=(input_dim,)),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.4),
        
        layers.Dense(128),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.3),
        
        layers.Dense(64),
        layers.BatchNormalization(),
        layers.Activation('relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

# =============================================================================
# MODEL 3: CLASS-WEIGHTED NETWORK (FOR IMBALANCED DATA)
# =============================================================================

def create_balanced_nn(input_dim, class_weights, learning_rate=0.001):
    """
    Network designed for imbalanced classes with focal loss
    """
    model = models.Sequential([
        layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        layers.Dropout(0.4),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    # Use class weights to handle imbalance
    model.compile(
        optimizer=optimizers.Adam(learning_rate=learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model, class_weights

# =============================================================================
# TRAINING FUNCTION WITH CALLBACKS
# =============================================================================

def train_neural_network(model, X_train, y_train, X_val, y_val, 
                         class_weights=None, epochs=100, batch_size=128):
    """
    Train neural network with proper callbacks and monitoring
    """
    print("=== TRAINING NEURAL NETWORK ===")
    
    # Callbacks for better training
    callbacks_list = [
        # Early stopping to prevent overfitting
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True
        ),
        
        # Reduce learning rate when stuck
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=8,
            min_lr=0.0001
        )
    ]
    
    # Train the model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks_list,
        class_weight=class_weights,
        verbose=1
    )
    
    return history

# =============================================================================
# EVALUATION FUNCTION
# =============================================================================

def evaluate_neural_network(model, X_test, y_test, model_name="Neural Network"):
    """
    Comprehensive evaluation of neural network performance
    """
    print(f"=== EVALUATING {model_name.upper()} ===")
    
    # Get predictions
    y_pred_probs = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_probs)
    
    # Confusion matrix for detailed analysis
    cm = confusion_matrix(y_test, y_pred)
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    total_actual_pits = np.sum(y_test)
    
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Pits Caught: {true_positives}/{total_actual_pits} ({100*true_positives/total_actual_pits:.1f}%)")
    print(f"False Alarms: {false_positives}")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    
    return y_pred_probs, y_pred, f1, roc_auc

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

def optimize_nn_hyperparameters(X_train, y_train, X_val, y_val, input_dim):
    """
    Simple grid search for neural network hyperparameters
    """
    print("=== OPTIMIZING HYPERPARAMETERS ===")
    
    # Parameters to test
    learning_rates = [0.001, 0.01, 0.0001]
    batch_sizes = [64, 128, 256]
    dropout_rates = [0.2, 0.3, 0.4]
    
    best_score = 0
    best_params = {}
    
    for lr in learning_rates:
        for batch_size in batch_sizes:
            for dropout in dropout_rates:
                print(f"Testing LR: {lr}, Batch: {batch_size}, Dropout: {dropout}")
                
                # Create model with these parameters
                model = models.Sequential([
                    layers.Dense(128, activation='relu', input_shape=(input_dim,)),
                    layers.Dropout(dropout),
                    layers.Dense(64, activation='relu'),
                    layers.Dropout(dropout),
                    layers.Dense(1, activation='sigmoid')
                ])
                
                model.compile(
                    optimizer=optimizers.Adam(learning_rate=lr),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Train with early stopping
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=50,  # Shorter for hyperparameter search
                    batch_size=batch_size,
                    verbose=0,
                    callbacks=[callbacks.EarlyStopping(patience=5)]
                )
                
                # Evaluate
                val_loss = min(history.history['val_loss'])
                if val_loss < best_score or best_score == 0:
                    best_score = val_loss
                    best_params = {
                        'learning_rate': lr,
                        'batch_size': batch_size,
                        'dropout_rate': dropout
                    }
                
                # Clean up
                del model
    
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_score:.4f}")
    
    return best_params

# =============================================================================
# MAIN NEURAL NETWORK PIPELINE
# =============================================================================

def run_neural_network_comparison(X_train, X_test, y_train, y_test):
    """
    Complete neural network pipeline comparing different architectures
    """
    print("=" * 60)
    print("F1 PIT STRATEGY - NEURAL NETWORK COMPARISON")
    print("=" * 60)
    
    # Prepare data
    X_train_scaled, X_test_scaled, y_train, y_test, scaler = prepare_data_for_nn(
        X_train, X_test, y_train, y_test
    )
    
    # Split training data for validation
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_scaled, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    input_dim = X_train_scaled.shape[1]
    
    # Calculate class weights for imbalanced data
    class_weight = {
        0: 1.0,
        1: len(y_train) / (2 * np.sum(y_train))  # Boost minority class
    }
    
    results = {}
    
    # Model 1: Simple Network
    print("Training Simple Neural Network...")
    model_simple = create_simple_nn(input_dim)
    history_simple = train_neural_network(
        model_simple, X_train_split, y_train_split, X_val_split, y_val_split
    )
    _, _, f1_simple, auc_simple = evaluate_neural_network(
        model_simple, X_test_scaled, y_test, "Simple NN"
    )
    results['Simple NN'] = (f1_simple, auc_simple)
    
    # Model 2: Deep Network
    print("Training Deep Neural Network...")
    model_deep = create_deep_nn(input_dim)
    history_deep = train_neural_network(
        model_deep, X_train_split, y_train_split, X_val_split, y_val_split
    )
    _, _, f1_deep, auc_deep = evaluate_neural_network(
        model_deep, X_test_scaled, y_test, "Deep NN"
    )
    results['Deep NN'] = (f1_deep, auc_deep)
    
    # Model 3: Balanced Network
    print("Training Class-Weighted Neural Network...")
    model_balanced, class_weights = create_balanced_nn(input_dim, class_weight)
    history_balanced = train_neural_network(
        model_balanced, X_train_split, y_train_split, X_val_split, y_val_split,
        class_weights=class_weights
    )
    _, _, f1_balanced, auc_balanced = evaluate_neural_network(
        model_balanced, X_test_scaled, y_test, "Balanced NN"
    )
    results['Balanced NN'] = (f1_balanced, auc_balanced)
    
    # Summary comparison
    print("=" * 60)
    print("NEURAL NETWORK SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 35)
    for model_name, (f1, auc) in results.items():
        print(f"{model_name:<15} {f1:<10.3f} {auc:<10.3f}")
    
    # Compare with your best tree-based model
    print()
    print("COMPARISON WITH TREE MODELS:")
    print(f"{'Random Forest':<15} {'0.782':<10} {'0.918':<10}")
    print(f"{'XGBoost':<15} {'0.766':<10} {'0.928':<10}")
    
    return results, scaler

