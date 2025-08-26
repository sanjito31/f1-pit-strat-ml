from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.over_sampling import RandomOverSampler, ADASYN
import joblib



def prepare():

    ## Read in the combined data file
    df_all = pd.read_parquet("data/all_processed.parquet")

    ## Split data into training (years 2019-2023) and validation (year 2024)
    train_data = df_all[df_all["Year"].isin([2019, 2020, 2021, 2022, 2023])]
    validation_data = df_all[df_all["Year"] == 2024]

    ## Remove Extraneous Data
    train_data = train_data.drop(["Year", "ShouldPitNext", "LapNumber"], axis=1)
    validation_data = validation_data.drop(["Year", "ShouldPitNext", "LapNumber"], axis=1)
    
    # X_train = features to train on, y_train = target value to train on
    X_train = train_data.drop(["PitWindow"], axis=1)
    y_train = train_data["PitWindow"]

    # X_val = features to validate on, y_val = target value to validate on
    X_val = validation_data.drop(["PitWindow"], axis=1)
    y_val = validation_data["PitWindow"]


    return X_train, y_train, X_val, y_val


# Comprehensive Evaluation Function
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

def comparison_report(all_models, y_val):

    # Compare All Methods
    print("\n" + "="*80)
    print("COMPARISON OF ALL METHODS ON 2024 SEASON")
    print("="*80)

    results = {}
    for k, v in all_models.items():
        results[k] = evaluate_model(y_val, v["pred"], v["proba"], k)

    # Summary Comparison
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

def feature_importance(model, X_train):
    # Step 10: Feature Importance (using best model)
    best_model = model  # Choose based on your preference
    print(f"\n=== FEATURE IMPORTANCE (Best Model) ===")
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': best_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(15))

def save_models(all_models, X_train):
    # Save the best models
    
    print("\n=== SAVING MODELS ===")
    
    for k,v in all_models.items():
        joblib.dump(v["model"], f'models/f1_{k}_model.pkl')

    joblib.dump(list(X_train.columns), 'models/f1_feature_names.pkl')

    print("All balanced models saved!")


def train_model(model_name: str, params, X_train, y_train, X_val):

    if model_name == "RandomForest":
        model = RandomForestClassifier(**params)
    elif model_name == "XGBoost":
        model = xgb.XGBClassifier(**params)
    else:
        raise Exception("Model must be 'RandomForest' or 'XGBoost'.")

    model.fit(X_train, y_train)
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]

    return {
        "model": model,
        "pred": pred,
        "proba": proba
    }



def main():

    X_train, y_train, X_val, y_val = prepare()


    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Original class distribution: {y_train.value_counts(normalize=True)}")

    all_models = {}


    ####### Original Model #######

    print("\n" + "="*60)
    print("CONTROL: ORIGINAL MODELS")
    print("="*60)

    rf_params = {'n_estimators': 100, 'random_state': 42, 'n_jobs': 1}
    xgb_params = {'random_state': 42, 'n_jobs': -1}

    print("Training Random Forest...")
    all_models["RF_Original"] = train_model("RandomForest", rf_params, X_train, y_train, X_val)

    print("Training XGBoost...")
    all_models["XGB_Original"] = train_model("XGBoost", xgb_params, X_train, y_train, X_val)

    
    ######## Class Balancing ########

    # Method 1: Class Weights (built into algorithms)
    print("\n" + "="*60)
    print("METHOD 1: CLASS WEIGHTS")
    print("="*60)

    # Calculate class weight ratio
    class_ratio = (y_train == False).sum() / (y_train == True).sum()
    print(f"Class ratio (False:True): {class_ratio:.1f}:1")

    rf_balanced_params = {'n_estimators': 100, 'class_weight': 'balanced', 'random_state': 42, 'n_jobs': 1}
    xgb_balanced_params = {'scale_pos_weight': class_ratio, 'random_state': 42, 'n_jobs': -1}

    print("Training Random Forest...")
    all_models["RF_Balanced"] = train_model("RandomForest", rf_balanced_params, X_train, y_train, X_val)

    print("Training XGBoost...")
    all_models["XGB_Balanced"] = train_model("XGBoost", xgb_balanced_params, X_train, y_train, X_val)


    # Method 2: SMOTE (Synthetic Minority Oversampling)
    print("\n" + "="*60)
    print("METHOD 2: SMOTE OVERSAMPLING")
    print("="*60)

    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    print(f"After SMOTE: {pd.Series(y_train_smote).value_counts(normalize=True)}")

    print("Training Random Forest...")
    all_models["RF_SMOTE"] = train_model("RandomForest", rf_params, X_train_smote, y_train_smote, X_val)

    print("Training XGBoost...")
    all_models["XGB_SMOTE"] = train_model("XGBoost", xgb_params, X_train_smote, y_train_smote, X_val)

    # Method 3: SMOTE-Tomek
    print("\n" + "="*60)
    print("METHOD 3: SMOTE-Tomek")
    print("="*60)

    smote_tomek = SMOTETomek(random_state=42)
    X_train_smote_tomek, y_train_smote_tomek = smote_tomek.fit_resample(X_train, y_train)

    print(f"After SMOTE-Tomek: {pd.Series(y_train_smote_tomek).value_counts(normalize=True)}")

    print("Training Random Forest...")
    all_models["RF_SMOTE_Tomek"] = train_model("RandomForest", rf_params, X_train_smote_tomek, y_train_smote_tomek, X_val)

    print("Training XGBoost...")
    all_models["XGB_SMOTE_Tomek"] = train_model("XGBoost", xgb_params, X_train_smote_tomek, y_train_smote_tomek, X_val)



    # Method 4: Random Undersamping
    print("\n" + "="*60)
    print("METHOD 4: Random Undersamping")
    print("="*60)

    rus = RandomUnderSampler(random_state=42)
    X_train_rus, y_train_rus = rus.fit_resample(X_train, y_train)

    print("Training Random Forest...")
    all_models["RF_RUS"] = train_model("RandomForest", rf_params, X_train_rus, y_train_rus, X_val)

    print("Training XGBoost...")
    all_models["XGB_RUS"] = train_model("XGBoost", xgb_params, X_train_rus, y_train_rus, X_val)




    # Method 5: Random Oversampling
    print("\n" + "="*60)
    print("METHOD 5: Random Oversampling")
    print("="*60)

    ros = RandomOverSampler(random_state=42)
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    print("Training Random Forest...")
    all_models["RF_ROS"] = train_model("RandomForest", rf_params, X_train_ros, y_train_ros, X_val)

    print("Training XGBoost...")
    all_models["XGB_ROS"] = train_model("XGBoost", xgb_params, X_train_ros, y_train_ros, X_val)


    # Method 6: ADASYN
    print("\n" + "="*60)
    print("METHOD 6: ADASYN")
    print("="*60)

    adasyn = ADASYN(random_state=42)
    X_train_adasyn, y_train_adasyn = adasyn.fit_resample(X_train, y_train)

    print("Training Random Forest...")
    all_models["RF_ADASYN"] = train_model("RandomForest", rf_params, X_train_adasyn, y_train_adasyn, X_val)

    print("Training XGBoost...")
    all_models["XGB_ADASYN"] = train_model("XGBoost", xgb_params, X_train_adasyn, y_train_adasyn, X_val)


    # Method 7: Tomek-Links
    print("\n" + "="*60)
    print("METHOD 7: Tomek-Links")
    print("="*60)

    tomek_links = TomekLinks()
    X_train_tomek_links, y_train_tomek_links = tomek_links.fit_resample(X_train, y_train)

    print("Training Random Forest...")
    all_models["RF_Tomek_Links"] = train_model("RandomForest", rf_params, X_train_tomek_links, y_train_tomek_links, X_val)

    print("Training XGBoost...")
    all_models["XGB_Tomek_Links"] = train_model("XGBoost", xgb_params, X_train_tomek_links, y_train_tomek_links, X_val)


    
    comparison_report(all_models, y_val)

    # feature_importance(all_models["RandomForest_Original"]["model"], X_train)

    save_models(all_models, X_train)




if __name__ == "__main__":
    main()

    # X_train, y_train, X_val, y_val = prepare()

    # best_model = joblib.load("models/f1_rf_balanced_model.pkl")

    # # with open("models/f1_rf_balanced_model.pkl", "rb") as f:
    # #     best_model = pickle.load(f)

    # feature_importance(best_model, X_train)
