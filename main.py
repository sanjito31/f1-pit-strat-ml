from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score, precision_recall_curve
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline



def main():

    ## Read in the combined data file
    # path = Path("data/all_processed.parquet")
    df_all = pd.read_parquet("data/all_processed.parquet")

    ## Split data into training (years 2019-2023) and validation (year 2024)
    train_data = df_all[df_all["Year"].isin([2019, 2020, 2021, 2022, 2023])]
    validation_data = df_all[df_all["Year"] == 2024]

    # X_train = features to train on, y_train = target value to train on
    X_train = train_data.drop(["Year", "RaceName", "ShouldPitNext"], axis=1)
    y_train = train_data["ShouldPitNext"]

    # X_val = features to validate on, y_val = target value to validate on
    X_val = validation_data.drop(["Year", "RaceName", "ShouldPitNext"], axis=1)
    y_val = validation_data["ShouldPitNext"]

    # Fix data types for SMOTE compatibility
    print(f"Original y_train dtype: {y_train.dtype}")
    print(f"Original y_train unique values: {y_train.unique()}")

    # Convert target to proper boolean/integer format
    y_train = y_train.astype(bool).astype(int)  # Convert to 0/1 integers
    y_val = y_val.astype(bool).astype(int)

    print(f"Converted y_train dtype: {y_train.dtype}")
    print(f"Converted y_train unique values: {y_train.unique()}")

    # Also ensure X_train has no non-numeric columns that could cause issues
    print(f"X_train dtypes:")
    print(X_train.dtypes.value_counts())

    # Convert ALL boolean and object columns to standard numeric types
    print("Converting all boolean/object columns to numeric...")
    for col in X_train.columns:
        if X_train[col].dtype in ['object', 'bool', 'boolean']:
            # Convert boolean to int (True->1, False->0)
            X_train[col] = X_train[col].astype(int)
            X_val[col] = X_val[col].astype(int)

    # Fill any NaN values that might have been created
    X_train = X_train.fillna(0)
    X_val = X_val.fillna(0)

    # Verify all columns are now numeric
    print(f"After conversion - X_train dtypes:")
    print(X_train.dtypes.value_counts())
    print(f"Sample of converted data:")
    print(X_train.head(2))

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Original class distribution: {y_train.value_counts(normalize=True)}")

    # Step 5: Class Balancing Techniques

    # Method 1: Class Weights (built into algorithms)
    print("\n" + "="*60)
    print("METHOD 1: CLASS WEIGHTS")
    print("="*60)

    # Calculate class weight ratio
    class_ratio = (y_train == False).sum() / (y_train == True).sum()
    print(f"Class ratio (False:True): {class_ratio:.1f}:1")

    # Random Forest with class weights
    rf_balanced = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced',  # Automatically balances classes
        random_state=42, 
        n_jobs=-1
    )
    rf_balanced.fit(X_train, y_train)
    rf_balanced_pred = rf_balanced.predict(X_val)
    rf_balanced_proba = rf_balanced.predict_proba(X_val)[:, 1]

    # XGBoost with scale_pos_weight
    xgb_balanced = xgb.XGBClassifier(
        scale_pos_weight=class_ratio,  # Weight positive class more heavily
        random_state=42,
        n_jobs=-1
    )
    xgb_balanced.fit(X_train, y_train)
    xgb_balanced_pred = xgb_balanced.predict(X_val)
    xgb_balanced_proba = xgb_balanced.predict_proba(X_val)[:, 1]

    # Method 2: SMOTE (Synthetic Minority Oversampling)
    print("\n" + "="*60)
    print("METHOD 2: SMOTE OVERSAMPLING")
    print("="*60)

    # Create SMOTE pipeline
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {pd.Series(y_train_smote).value_counts(normalize=True)}")

    # Train models on SMOTE data
    rf_smote = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_smote.fit(X_train_smote, y_train_smote)
    rf_smote_pred = rf_smote.predict(X_val)
    rf_smote_proba = rf_smote.predict_proba(X_val)[:, 1]

    xgb_smote = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    xgb_smote.fit(X_train_smote, y_train_smote)
    xgb_smote_pred = xgb_smote.predict(X_val)
    xgb_smote_proba = xgb_smote.predict_proba(X_val)[:, 1]

    # Step 6: Comprehensive Evaluation Function
    def evaluate_model(y_true, y_pred, y_proba, model_name):
        print(f"\n=== {model_name} RESULTS ===")
        print(f"Accuracy: {accuracy_score(y_true, y_pred):.3f}")
        print(f"F1-Score: {f1_score(y_true, y_pred):.3f}")
        print(f"ROC-AUC: {roc_auc_score(y_true, y_proba):.3f}")
        
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=['No Pit', 'Pit']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_true, y_pred)
        print(cm)
        
        # Calculate specific metrics
        tn, fp, fn, tp = cm.ravel()
        print(f"True Positives (Correctly predicted pits): {tp}")
        print(f"False Negatives (Missed pit opportunities): {fn}")
        print(f"False Positives (False pit alarms): {fp}")
        
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred),
            'auc': roc_auc_score(y_true, y_proba),
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn
        }

    # Step 7: Compare All Methods
    print("\n" + "="*80)
    print("COMPARISON OF ALL METHODS ON 2024 SEASON")
    print("="*80)

    results = {}
    results['RF_Original'] = evaluate_model(y_val, rf_balanced_pred, rf_balanced_proba, 'RANDOM FOREST (Original)')
    results['RF_Balanced'] = evaluate_model(y_val, rf_balanced_pred, rf_balanced_proba, 'RANDOM FOREST (Class Weights)')
    results['RF_SMOTE'] = evaluate_model(y_val, rf_smote_pred, rf_smote_proba, 'RANDOM FOREST (SMOTE)')
    results['XGB_Balanced'] = evaluate_model(y_val, xgb_balanced_pred, xgb_balanced_proba, 'XGBOOST (Scale Pos Weight)')
    results['XGB_SMOTE'] = evaluate_model(y_val, xgb_smote_pred, xgb_smote_proba, 'XGBOOST (SMOTE)')

    # Step 8: Summary Comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    comparison_df = pd.DataFrame({
        model: {
            'F1-Score': f"{metrics['f1']:.3f}",
            'ROC-AUC': f"{metrics['auc']:.3f}",
            'Pits Caught': f"{metrics['tp']}/{metrics['tp'] + metrics['fn']} ({metrics['tp']/(metrics['tp'] + metrics['fn'])*100:.1f}%)",
            'False Alarms': f"{metrics['fp']}"
        }
        for model, metrics in results.items()
    }).T

    print(comparison_df)

    # Step 9: Feature Importance (using best model)
    best_model = xgb_balanced  # Choose based on your preference
    print(f"\n=== FEATURE IMPORTANCE (Best Model) ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(15))

    # Step 10: Save the best models
    import joblib
    print("\n=== SAVING MODELS ===")
    joblib.dump(rf_balanced, 'models/f1_rf_balanced_model.pkl')
    joblib.dump(xgb_balanced, 'models/f1_xgb_balanced_model.pkl')
    joblib.dump(rf_smote, 'models/f1_rf_smote_model.pkl')
    joblib.dump(xgb_smote, 'models/f1_xgb_smote_model.pkl')
    joblib.dump(list(X_train.columns), 'models/f1_feature_names.pkl')

    print("All balanced models saved!")
    print("\nRecommendation: Choose the model with the best F1-Score and ROC-AUC for your use case.")






if __name__ == "__main__":
    main()
