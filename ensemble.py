import numpy as np
import pandas as pd
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score

# =============================================================================
# METHOD 1: SIMPLE AVERAGING ENSEMBLE
# =============================================================================

def simple_ensemble(rf_model, xgb_model, X_test, y_test, threshold=0.5):
    """
    Simple averaging ensemble of RF and XGBoost predictions
    """
    print("=== SIMPLE AVERAGING ENSEMBLE ===")
    
    # Get probability predictions from both models
    rf_probs = rf_model.predict_proba(X_test)[:, 1]  # Probability of pit window
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Average the probabilities
    ensemble_probs = (rf_probs + xgb_probs) / 2
    
    # Make final predictions using threshold
    ensemble_preds = (ensemble_probs >= threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_test, ensemble_preds)
    roc_auc = roc_auc_score(y_test, ensemble_probs)
    
    # Count true positives and false positives
    cm = confusion_matrix(y_test, ensemble_preds)
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    total_actual_pits = np.sum(y_test)
    
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Pits Caught: {true_positives}/{total_actual_pits} ({100*true_positives/total_actual_pits:.1f}%)")
    print(f"False Alarms: {false_positives}")
    print()
    
    return ensemble_probs, ensemble_preds, f1, roc_auc

# =============================================================================
# METHOD 2: WEIGHTED ENSEMBLE
# =============================================================================

def weighted_ensemble(rf_model, xgb_model, X_test, y_test, rf_weight=0.6, threshold=0.5):
    """
    Weighted ensemble giving different weights to each model
    rf_weight: weight for Random Forest (XGBoost gets 1-rf_weight)
    """
    print("=== WEIGHTED ENSEMBLE ===")
    print(f"RF Weight: {rf_weight}, XGB Weight: {1-rf_weight}")
    
    # Get probability predictions
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    
    # Weighted average
    ensemble_probs = (rf_weight * rf_probs) + ((1-rf_weight) * xgb_probs)
    ensemble_preds = (ensemble_probs >= threshold).astype(int)
    
    # Calculate metrics
    f1 = f1_score(y_test, ensemble_preds)
    roc_auc = roc_auc_score(y_test, ensemble_probs)
    
    # Count results
    cm = confusion_matrix(y_test, ensemble_preds)
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    total_actual_pits = np.sum(y_test)
    
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Pits Caught: {true_positives}/{total_actual_pits} ({100*true_positives/total_actual_pits:.1f}%)")
    print(f"False Alarms: {false_positives}")
    print()
    
    return ensemble_probs, ensemble_preds, f1, roc_auc

# =============================================================================
# METHOD 3: SKLEARN VOTING CLASSIFIER
# =============================================================================

def voting_ensemble(rf_model, xgb_model, X_train, y_train, X_test, y_test):
    """
    Sklearn's VotingClassifier for more robust ensemble
    """
    print("=== VOTING CLASSIFIER ENSEMBLE ===")
    
    # Create voting ensemble
    voting_clf = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('xgb', xgb_model)
        ],
        voting='soft'  # Use probabilities instead of hard predictions
    )
    
    # Fit the ensemble (this refits the individual models)
    voting_clf.fit(X_train, y_train)
    
    # Make predictions
    ensemble_preds = voting_clf.predict(X_test)
    ensemble_probs = voting_clf.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    f1 = f1_score(y_test, ensemble_preds)
    roc_auc = roc_auc_score(y_test, ensemble_probs)
    
    # Count results
    cm = confusion_matrix(y_test, ensemble_preds)
    true_positives = cm[1, 1]
    false_positives = cm[0, 1]
    total_actual_pits = np.sum(y_test)
    
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Pits Caught: {true_positives}/{total_actual_pits} ({100*true_positives/total_actual_pits:.1f}%)")
    print(f"False Alarms: {false_positives}")
    print()
    
    return voting_clf, ensemble_probs, ensemble_preds, f1, roc_auc

# =============================================================================
# METHOD 4: OPTIMIZE WEIGHTS
# =============================================================================

def find_optimal_weights(rf_model, xgb_model, X_val, y_val):
    """
    Find optimal weights by testing different combinations
    """
    print("=== FINDING OPTIMAL WEIGHTS ===")
    
    best_f1 = 0
    best_weight = 0.5
    
    # Test different RF weights from 0.1 to 0.9
    weights_to_test = np.arange(0.1, 1.0, 0.1)
    
    rf_probs = rf_model.predict_proba(X_val)[:, 1]
    xgb_probs = xgb_model.predict_proba(X_val)[:, 1]
    
    for rf_weight in weights_to_test:
        xgb_weight = 1 - rf_weight
        
        # Create weighted ensemble
        ensemble_probs = (rf_weight * rf_probs) + (xgb_weight * xgb_probs)
        ensemble_preds = (ensemble_probs >= 0.5).astype(int)
        
        # Calculate F1 score
        f1 = f1_score(y_val, ensemble_preds)
        
        print(f"RF Weight: {rf_weight:.1f}, XGB Weight: {xgb_weight:.1f} -> F1: {f1:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_weight = rf_weight
    
    print(f"\nOptimal RF Weight: {best_weight:.1f}")
    print(f"Best F1 Score: {best_f1:.3f}")
    
    return best_weight

# =============================================================================
# MAIN ENSEMBLE COMPARISON FUNCTION
# =============================================================================

def compare_all_ensembles(rf_model, xgb_model, X_train, y_train, X_test, y_test):
    """
    Compare all ensemble methods and individual models
    """
    print("=" * 60)
    print("ENSEMBLE COMPARISON")
    print("=" * 60)
    
    # Individual model performance for comparison
    print("=== INDIVIDUAL MODEL PERFORMANCE ===")
    
    # Random Forest
    rf_preds = rf_model.predict(X_test)
    rf_probs = rf_model.predict_proba(X_test)[:, 1]
    rf_f1 = f1_score(y_test, rf_preds)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_cm = confusion_matrix(y_test, rf_preds)
    print(f"Random Forest - F1: {rf_f1:.3f}, AUC: {rf_auc:.3f}, False Alarms: {rf_cm[0,1]}")
    
    # XGBoost
    xgb_preds = xgb_model.predict(X_test)
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_f1 = f1_score(y_test, xgb_preds)
    xgb_auc = roc_auc_score(y_test, xgb_probs)
    xgb_cm = confusion_matrix(y_test, xgb_preds)
    print(f"XGBoost - F1: {xgb_f1:.3f}, AUC: {xgb_auc:.3f}, False Alarms: {xgb_cm[0,1]}")
    print()
    
    # Ensemble methods
    results = {}
    
    # Simple averaging
    _, _, f1_simple, auc_simple = simple_ensemble(rf_model, xgb_model, X_test, y_test)
    results['Simple Average'] = (f1_simple, auc_simple)
    
    # Weighted ensemble (favoring RF since it had better precision)
    _, _, f1_weighted, auc_weighted = weighted_ensemble(rf_model, xgb_model, X_test, y_test, rf_weight=0.6)
    results['Weighted (RF=0.6)'] = (f1_weighted, auc_weighted)
    
    # Voting classifier
    _, _, _, f1_voting, auc_voting = voting_ensemble(rf_model, xgb_model, X_train, y_train, X_test, y_test)
    results['Voting Classifier'] = (f1_voting, auc_voting)
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Method':<20} {'F1-Score':<10} {'ROC-AUC':<10}")
    print("-" * 40)
    print(f"{'Random Forest':<20} {rf_f1:<10.3f} {rf_auc:<10.3f}")
    print(f"{'XGBoost':<20} {xgb_f1:<10.3f} {xgb_auc:<10.3f}")
    for method, (f1, auc) in results.items():
        print(f"{method:<20} {f1:<10.3f} {auc:<10.3f}")
    
    return results