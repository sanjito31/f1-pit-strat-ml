from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import xgboost as xgb
import pandas as pd
import pathlib as Path


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

    # Random Forest baseline
    print("\nTraining Random Forest...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_val)

    # XGBoost
    print("Training XGBoost...")
    xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)

    # Step 7: Evaluate Results
    print("\n" + "="*50)
    print("RESULTS ON 2024 SEASON")
    print("="*50)

    print("\n=== RANDOM FOREST RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_val, rf_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, rf_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, rf_pred))

    print("\n=== XGBOOST RESULTS ===")
    print(f"Accuracy: {accuracy_score(y_val, xgb_pred):.3f}")
    print("\nClassification Report:")
    print(classification_report(y_val, xgb_pred))
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, xgb_pred))

    # Step 8: Feature Importance Analysis
    print("\n=== MOST IMPORTANT FEATURES (Random Forest) ===")
    feature_names = X_train.columns
    rf_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    print(rf_importance.head(15))

    print("\n=== MOST IMPORTANT FEATURES (XGBoost) ===")
    xgb_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': xgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(xgb_importance.head(15))

    # Step 9: Class distribution analysis
    print("\n=== DATA DISTRIBUTION ===")
    print("Training set class distribution:")
    print(y_train.value_counts(normalize=True))
    print("\nValidation set class distribution:")
    print(y_val.value_counts(normalize=True))

    print("\nTraining complete! Models ready for predictions.")

    import joblib

    print("\n=== SAVING MODELS ===")
    joblib.dump(rf, 'f1_random_forest_model.pkl')
    joblib.dump(xgb_model, 'f1_xgboost_model.pkl')
    joblib.dump(list(X_train.columns), 'f1_feature_names.pkl')

    print("Models saved as:")
    print("- f1_random_forest_model.pkl")
    print("- f1_xgboost_model.pkl") 
    print("- f1_feature_names.pkl")
    print("\nTraining complete! Models ready for predictions.")






if __name__ == "__main__":
    main()
