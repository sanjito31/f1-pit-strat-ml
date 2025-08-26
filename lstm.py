import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, callbacks, optimizers
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow
np.random.seed(42)
tf.random.set_seed(42)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# =============================================================================
# SEQUENCE CREATION FOR F1 RACE STRUCTURE
# =============================================================================

def identify_columns(df):
    """
    Identify different types of columns in your F1 dataset
    """
    print("=== IDENTIFYING COLUMN TYPES ===")
    print(f"All columns: {list(df.columns)}")
    
    # Find dummy encoded columns
    driver_cols = [col for col in df.columns if col.startswith('Driver_')]
    race_cols = [col for col in df.columns if col.startswith('Race_')]
    team_cols = [col for col in df.columns if col.startswith('Team_')]
    
    # Identify key columns - be more flexible with naming
    lap_col = None
    for possible_lap_col in ['LapNumber', 'Lap', 'lap', 'LAP', 'lap_number']:
        if possible_lap_col in df.columns:
            lap_col = possible_lap_col
            break
    
    target_col = None
    for possible_target_col in ['PitWindow', 'pit_window', 'ShouldPit', 'should_pit']:
        if possible_target_col in df.columns:
            target_col = possible_target_col
            break
    
    # Feature columns (everything except identifiers and target)
    exclude_cols = driver_cols + race_cols + team_cols + ([lap_col] if lap_col else []) + ([target_col] if target_col else [])
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    print(f"Driver columns: {len(driver_cols)} (e.g., {driver_cols[:3] if driver_cols else 'None found'})")
    print(f"Race columns: {len(race_cols)} (e.g., {race_cols[:3] if race_cols else 'None found'})")
    print(f"Team columns: {len(team_cols)} (e.g., {team_cols[:3] if team_cols else 'None found'})")
    print(f"Feature columns: {len(feature_cols)} (first 5: {feature_cols[:5] if feature_cols else 'None found'})")
    print(f"Lap column: {lap_col}")
    print(f"Target column: {target_col}")
    
    # Error checking
    if lap_col is None:
        raise ValueError("Could not find lap column. Expected one of: 'LapNumber', 'Lap', 'lap', 'LAP', 'lap_number'")
    if target_col is None:
        raise ValueError("Could not find target column. Expected one of: 'PitWindow', 'pit_window', 'ShouldPit', 'should_pit'")
    if not driver_cols:
        print("WARNING: No driver columns found (expected Driver_HAM, Driver_VER, etc.)")
    if not race_cols:
        print("WARNING: No race columns found (expected Race_monaco, Race_silverstone, etc.)")
    
    print()
    
    return {
        'driver_cols': driver_cols,
        'race_cols': race_cols, 
        'team_cols': team_cols,
        'feature_cols': feature_cols,
        'lap_col': lap_col,
        'target_col': target_col
    }

def create_race_driver_sequences(df, sequence_length=5, min_stint_length=None):
    """
    Create sequences respecting F1 race and driver boundaries
    Each sequence comes from the same driver in the same race
    """
    print(f"=== CREATING RACE-AWARE SEQUENCES (Length: {sequence_length}) ===")
    
    if min_stint_length is None:
        min_stint_length = sequence_length + 2  # Need at least seq_length + some laps
    
    # Identify column structure
    col_info = identify_columns(df)
    
    # Get active driver and race for each lap (from dummy encoding)
    df_work = df.copy()
    
    # Find which driver is active (has 1 in driver columns)
    driver_mapping = {}
    for idx, row in df_work.iterrows():
        for driver_col in col_info['driver_cols']:
            if row[driver_col] == 1:
                driver_mapping[idx] = driver_col
                break
    
    # Find which race is active  
    race_mapping = {}
    for idx, row in df_work.iterrows():
        for race_col in col_info['race_cols']:
            if row[race_col] == 1:
                race_mapping[idx] = race_col
                break
    
    df_work['ActiveDriver'] = df_work.index.map(driver_mapping)
    df_work['ActiveRace'] = df_work.index.map(race_mapping)
    
    print(f"Found {len(set(driver_mapping.values()))} active drivers")
    print(f"Found {len(set(race_mapping.values()))} active races")
    
    # Create sequences
    sequences = []
    targets = []
    metadata = []  # Track which race/driver each sequence comes from
    
    # Group by race and driver combination
    grouped = df_work.groupby(['ActiveRace', 'ActiveDriver'])
    
    processed_groups = 0
    for (race, driver), group_data in grouped:
        if len(group_data) < min_stint_length:
            continue  # Skip short stints
        
        # Sort by lap number to ensure temporal order
        group_data = group_data.sort_values(col_info['lap_col'])
        
        # Create sequences for this race/driver combination
        group_sequences = 0
        for i in range(sequence_length, len(group_data)):
            # Get sequence of previous laps (features only)
            seq_data = group_data.iloc[i-sequence_length:i]
            sequence = seq_data[col_info['feature_cols']].values
            
            # Get target for current lap
            target = group_data.iloc[i][col_info['target_col']]
            
            # Store sequence info
            sequences.append(sequence)
            targets.append(target)
            metadata.append({
                'race': race,
                'driver': driver, 
                'start_lap': seq_data.iloc[0][col_info['lap_col']],
                'target_lap': group_data.iloc[i][col_info['lap_col']]
            })
            
            group_sequences += 1
        
        processed_groups += 1
        if processed_groups % 50 == 0:
            print(f"Processed {processed_groups} race/driver combinations...")
    
    X_sequences = np.array(sequences)
    y_sequences = np.array(targets)
    
    print(f"Created {len(sequences)} sequences from {processed_groups} race/driver combinations")
    print(f"Sequence shape: {X_sequences.shape}")
    print(f"Average sequences per race/driver: {len(sequences)/processed_groups:.1f}")
    print(f"Target distribution: {np.bincount(y_sequences)}")
    print()
    
    return X_sequences, y_sequences, metadata, col_info

# =============================================================================
# ALTERNATIVE: SIMPLER APPROACH WITHOUT DUMMY DECODING
# =============================================================================

def create_sequences_simple_approach(df, sequence_length=5, race_col='Race', driver_col='Driver', lap_col='Lap'):
    """
    Simpler approach if you can provide the actual race/driver column names
    This assumes you have columns like 'Race_monaco' where the race name is clear
    """
    print(f"=== CREATING SEQUENCES - SIMPLE APPROACH ===")
    
    col_info = identify_columns(df)
    
    # If you have a way to identify unique race/driver combinations more easily
    sequences = []
    targets = []
    
    # Group by unique patterns in dummy encoding
    # This creates a unique identifier for each race/driver combo
    grouping_cols = col_info['driver_cols'] + col_info['race_cols']
    df_grouped = df.groupby(grouping_cols)
    
    processed = 0
    for group_key, group_data in df_grouped:
        if len(group_data) < sequence_length + 2:
            continue
        
        # Sort by lap
        group_data = group_data.sort_values(col_info['lap_col'])
        
        # Create sequences
        for i in range(sequence_length, len(group_data)):
            sequence = group_data.iloc[i-sequence_length:i][col_info['feature_cols']].values
            target = group_data.iloc[i][col_info['target_col']]
            
            sequences.append(sequence)
            targets.append(target)
        
        processed += 1
        if processed % 100 == 0:
            print(f"Processed {processed} groups...")
    
    return np.array(sequences), np.array(targets), col_info

# =============================================================================
# MANUAL RACE/DRIVER SPECIFICATION (RECOMMENDED)
# =============================================================================

def create_sequences_manual_mapping(df, race_driver_mapping, sequence_length=5):
    """
    Most reliable approach: you provide the race/driver mapping manually
    
    Parameters:
    race_driver_mapping: dict like {
        'monaco_hamilton': df[(df['Race_monaco']==1) & (df['Driver_HAM']==1)].index,
        'monaco_verstappen': df[(df['Race_monaco']==1) & (df['Driver_VER']==1)].index,
        ...
    }
    """
    print(f"=== CREATING SEQUENCES - MANUAL MAPPING ===")
    
    col_info = identify_columns(df)
    
    sequences = []
    targets = []
    
    for combo_name, indices in race_driver_mapping.items():
        combo_data = df.loc[indices].sort_values(col_info['lap_col'])
        
        if len(combo_data) < sequence_length + 2:
            print(f"Skipping {combo_name}: only {len(combo_data)} laps")
            continue
        
        # Create sequences for this race/driver combo
        combo_sequences = 0
        for i in range(sequence_length, len(combo_data)):
            sequence = combo_data.iloc[i-sequence_length:i][col_info['feature_cols']].values
            target = combo_data.iloc[i][col_info['target_col']]
            
            sequences.append(sequence)
            targets.append(target)
            combo_sequences += 1
        
        print(f"{combo_name}: {combo_sequences} sequences from {len(combo_data)} laps")
    
    print(f"Total sequences created: {len(sequences)}")
    return np.array(sequences), np.array(targets), col_info

# =============================================================================
# HELPER FUNCTION TO CREATE RACE/DRIVER MAPPING
# =============================================================================

def create_race_driver_mapping(df, max_combinations=None):
    """
    Helper to create race/driver mapping from your dummy encoded data
    """
    print("=== CREATING RACE/DRIVER MAPPING ===")
    
    col_info = identify_columns(df)
    mapping = {}
    
    # Find all combinations
    for race_col in col_info['race_cols'][:5]:  # Limit for testing
        race_name = race_col.replace('Race_', '')
        race_data = df[df[race_col] == 1]
        
        for driver_col in col_info['driver_cols'][:10]:  # Limit for testing  
            driver_name = driver_col.replace('Driver_', '')
            combo_data = race_data[race_data[driver_col] == 1]
            
            if len(combo_data) >= 10:  # Only include if enough laps
                combo_name = f"{race_name}_{driver_name}"
                mapping[combo_name] = combo_data.index.tolist()
                
                if max_combinations and len(mapping) >= max_combinations:
                    break
        
        if max_combinations and len(mapping) >= max_combinations:
            break
    
    print(f"Created mapping for {len(mapping)} race/driver combinations")
    for combo, indices in list(mapping.items())[:5]:
        print(f"  {combo}: {len(indices)} laps")
    
    return mapping

# =============================================================================
# MAIN LSTM PIPELINE ADAPTED FOR YOUR DATA
# =============================================================================

def run_f1_lstm_with_structure(train_df, test_df, sequence_length=5):
    """
    Complete LSTM pipeline using your race/driver structure with proper time-based splitting
    
    Parameters:
    train_df: DataFrame with 2019-2023 data
    test_df: DataFrame with 2024 data  
    """
    print("=" * 60)
    print("F1 LSTM WITH TIME-BASED SPLITTING")
    print("=" * 60)
    
    print("Creating training sequences from 2019-2023...")
    # Create sequences from training data
    try:
        X_train_seq, y_train_seq, metadata_train, col_info = create_race_driver_sequences(
            train_df, sequence_length=sequence_length
        )
    except Exception as e:
        print(f"Automatic method failed: {e}")
        print("Falling back to manual mapping...")
        race_driver_mapping = create_race_driver_mapping(train_df, max_combinations=100)
        X_train_seq, y_train_seq, col_info = create_sequences_manual_mapping(
            train_df, race_driver_mapping, sequence_length=sequence_length
        )
    
    print("Creating test sequences from 2024...")
    # Create sequences from test data  
    try:
        X_test_seq, y_test_seq, metadata_test, _ = create_race_driver_sequences(
            test_df, sequence_length=sequence_length
        )
    except Exception as e:
        print("Falling back to manual mapping for test data...")
        race_driver_mapping_test = create_race_driver_mapping(test_df, max_combinations=50)
        X_test_seq, y_test_seq, _ = create_sequences_manual_mapping(
            test_df, race_driver_mapping_test, sequence_length=sequence_length
        )
    
    # Scale features using ONLY training data
    print("Scaling features using training data distribution...")
    n_train_samples, n_timesteps, n_features = X_train_seq.shape
    n_test_samples = X_test_seq.shape[0]
    
    # Fit scaler on training data only
    X_train_reshaped = X_train_seq.reshape(-1, n_features)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_reshaped)
    X_train_scaled = X_train_scaled.reshape(n_train_samples, n_timesteps, n_features)
    
    # Transform test data using training scaler
    X_test_reshaped = X_test_seq.reshape(-1, n_features)
    X_test_scaled = scaler.transform(X_test_reshaped)
    X_test_scaled = X_test_scaled.reshape(n_test_samples, n_timesteps, n_features)
    
    # Validation split from training data (use last 20% of training years)
    val_split_idx = int(0.8 * len(X_train_scaled))
    X_val = X_train_scaled[val_split_idx:]
    y_val = y_train_seq[val_split_idx:]
    X_train_final = X_train_scaled[:val_split_idx]
    y_train_final = y_train_seq[:val_split_idx]
    
    print(f"Training sequences: {len(X_train_final)} (2019-2022)")
    print(f"Validation sequences: {len(X_val)} (2023)")  
    print(f"Test sequences: {len(X_test_scaled)} (2024)")
    print(f"Training target distribution: {np.bincount(y_train_final)}")
    print(f"Test target distribution: {np.bincount(y_test_seq)}")
    
    # Create and train LSTM model
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(n_timesteps, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),
        layers.Dense(16, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    # Class weights based on training data only
    class_weight = {
        0: 1.0,
        1: len(y_train_final) / (2 * np.sum(y_train_final))
    }
    
    print("Training LSTM model...")
    print("Training on 2019-2022, validating on 2023, testing on 2024...")
    history = model.fit(
        X_train_final, y_train_final,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=64,
        class_weight=class_weight,
        callbacks=[
            callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
        ],
        verbose=1
    )
    
    # Evaluate on 2024 data
    y_pred_probs = model.predict(X_test_scaled, verbose=0).flatten()
    y_pred = (y_pred_probs >= 0.5).astype(int)
    
    f1 = f1_score(y_test_seq, y_pred)
    roc_auc = roc_auc_score(y_test_seq, y_pred_probs)
    
    cm = confusion_matrix(y_test_seq, y_pred)
    true_positives = cm[1, 1] if cm.shape == (2, 2) and len(cm) > 1 else 0
    false_positives = cm[0, 1] if cm.shape == (2, 2) and len(cm) > 1 else 0
    
    print("=" * 60)
    print("LSTM RESULTS (2024 VALIDATION)")
    print("=" * 60)
    print(f"F1-Score: {f1:.3f}")
    print(f"ROC-AUC: {roc_auc:.3f}")
    print(f"Pits Caught: {true_positives}/{np.sum(y_test_seq)} ({100*true_positives/max(1,np.sum(y_test_seq)):.1f}%)")
    print(f"False Alarms: {false_positives}")
    
    print("\nComparison with your tree models (same 2024 validation):")
    print("Random Forest: F1=0.782, AUC=0.918")
    print("XGBoost:       F1=0.766, AUC=0.928")
    
    return model, scaler, history, f1, roc_auc

# =============================================================================
# USAGE EXAMPLE WITH TIME-BASED SPLITTING
# =============================================================================

"""
USAGE WITH PROPER TIME-BASED SPLITTING:

# Method 1: Pre-split your data by year (RECOMMENDED)
train_df = df[df['year'].isin([2019, 2020, 2021, 2022, 2023])]  
test_df = df[df['year'] == 2024]

# Or if year is encoded in race names:
# train_df = df[~df.columns[df.columns.str.contains('2024')].any(axis=1)]
# test_df = df[df.columns[df.columns.str.contains('2024')].any(axis=1)]

# Run LSTM with proper time-based splitting
model, scaler, history, f1_score, roc_auc = run_f1_lstm_with_structure(
    train_df=train_df,  # 2019-2023 data
    test_df=test_df,    # 2024 data
    sequence_length=5
)

# Method 2: If you can't easily separate by year, pass the column name
# This assumes you have a 'Season' or 'Year' column
def split_by_time_column(df, time_col='Season', train_years=[2019,2020,2021,2022,2023], test_years=[2024]):
    train_df = df[df[time_col].isin(train_years)]
    test_df = df[df[time_col].isin(test_years)]
    return train_df, test_df

# train_df, test_df = split_by_time_column(df)
# model, scaler, history, f1, auc = run_f1_lstm_with_structure(train_df, test_df)
"""

def main():

    ## Read in the combined data file
    df_all = pd.read_parquet("data/all_processed.parquet")

    ## Dummy Encode Track
    teams = pd.get_dummies(df_all["RaceName"], prefix="Race")
    df_all = pd.concat([df_all, teams], axis=1)

    ## Split data into training (years 2019-2023) and validation (year 2024)
    train_data = df_all[df_all["Year"].isin([2019, 2020, 2021, 2022, 2023])]
    validation_data = df_all[df_all["Year"] == 2024]

    ## Remove Extraneous Data
    train_data = train_data.drop(["Year", "RaceName", "ShouldPitNext"], axis=1)
    validation_data = validation_data.drop(["Year", "RaceName", "ShouldPitNext"], axis=1)

    model, scaler, history, f1, auc = run_f1_lstm_with_structure(
        train_df=train_data,
        test_df=validation_data,
        sequence_length=5
    )

if __name__ == "__main__":
    main()

