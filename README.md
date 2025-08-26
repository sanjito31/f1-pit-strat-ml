# F1 Optimal Pit Stop Predictor

A machine learning project trained on F1 lap data to determine the optimal range of laps that a driver/team would make a pit stop. 

In F1, race strategy is crucial for gaining a competitive edge, with pit stop strategy playing a vital role. A well timed pit stop could be make or break for a race. Therefore, teams in F1 spend significant amounts of valuable resources on race strategists and engineers to interpret the copious amounts of timing, weather, and telemetry data to determine the optimal window for a driver to make a pit stop for maximal gain. This project aims to accomplish a similar task using machine learning models trained on F1 lap by lap race data from 2018 to 2024. 

## Overview
- Historical F1 lap data is retrieved via the FastAPI/Ergast API ```download_data.py```

- The data is then processed and key race metrics are engineered per lap per driver ```processor.py``` to understand tire degradation, which includes:
    - **Pace Drop Off** 
    - **3 Lap Moving Average Time**
    - **Lap Time Trend**
    - **Position Changes**
    - **Straight Line Speed Dropoff**
    - **Gap To Driver Ahead**
    - **Race Progress (Normalized)**
    - **Should Pit Next** (boolean determination)
    - **Weather Data** (air temp, track temp, humidity, rain, etc)

    - **Pit Window** <- What the model is trained to detect. (5 lap window of optimal pit timing. Pit lap is lap 3 of 5 in this window)
    Additional data such as safety car status, driver names, and team names are also encoded.

- **Random Forest** and **XGBoost** models are then trained using the following class imbalance methods to determine the best fit ```comparison.py```.
    - Class weights
    - Random Oversampling
    - SMOTE Oversampling
    - ADASYN Oversampling
    - Random Undersampling
    - Tomek-Link Undersampling
    - SMOTE-Tomek Over/Undersampling
- Using the best model, we then can apply it to current/real-time data via a convenient Streamlit platform (coming soon)

## Results

Overall, the **Random Forest Balanced** model seemed to be the best choice. This model was chosen primarily due to its low levels of **false alarms** at 253. Additional key metrics include a **high ROC-AUC** (receiving operator characteristic area under the curve) of 0.918 and good pit stop prediction success rate **67.3%**. Models with higher rates of catching pit stop windows were disregarded due to higher false alarm rates, which would cause disruption in real situations.

Looking at the feature importance, we can see that many of the engineered features are key decision factors for the model, which mirror real life pit stop decisions. 

```
============================================================
SUMMARY COMPARISON
============================================================
                F1-Score ROC-AUC        Pits Caught False Alarms
RF_Original        0.780   0.919  3428/5023 (68.2%)          336
XGB_Original       0.766   0.928  3637/5023 (72.4%)          838

** RF_Balanced     0.781   0.918  3380/5023 (67.3%)          253 **

XGB_Balanced       0.726   0.932  4125/5023 (82.1%)         2218
RF_SMOTE           0.756   0.916  3663/5023 (72.9%)         1002
XGB_SMOTE          0.662   0.907  3909/5023 (77.8%)         2876
RF_SMOTE_Tomek     0.750   0.915  3672/5023 (73.1%)         1093
XGB_SMOTE_Tomek    0.655   0.906  3882/5023 (77.3%)         2955
RF_RUS             0.742   0.926  3960/5023 (78.8%)         1698
XGB_RUS            0.707   0.929  4172/5023 (83.1%)         2611
RF_ROS             0.781   0.922  3481/5023 (69.3%)          409
XGB_ROS            0.703   0.927  4104/5023 (81.7%)         2552
RF_ADASYN          0.750   0.914  3728/5023 (74.2%)         1188
XGB_ADASYN         0.640   0.899  3917/5023 (78.0%)         3298
RF_Tomek_Links     0.774   0.917  3420/5023 (68.1%)          397
XGB_Tomek_Links    0.759   0.927  3673/5023 (73.1%)          988

=== FEATURE IMPORTANCE (RF_Balanced) ===
           feature  importance
55    RaceProgress    0.153660
1         TyreLife    0.145870
49     PaceDropoff    0.082043
51    LapTimeTrend    0.077063
4       LapTime(s)    0.046052
50      LapTimeMA3    0.045423
54   GapToAhead(s)    0.033431
60       TrackTemp    0.029795
57        Humidity    0.028430
53    SpeedDropoff    0.028374
52  PositionChange    0.028355
56         AirTemp    0.027506
61   WindDirection    0.025284
58        Pressure    0.024801
62       WindSpeed    0.022256
```

## Requirements
- Python version 3.12

## Setup and Use

1. Clone this repository
    ```
    git clone git@github.com:sanjito31/f1-pit-strat-ml.git
    ```

2. Install python version specified in ```runtime.txt``` 
    - Should be Python 3.12
    - This is due to the fact that Python 3.12 is the last supported version for TensorFlow

3. Set up python virtual environment
    ```
    $ python3.12 -m venv .venv # MAKE SURE VERSION MATCHES WHATS IN RUNTIME.TXT
    $ source .venv/bin/activate
    (.venv) $ 
    ```
4. Install dependencies
    ```
    (.venv) $ pip install -r requirements.txt
    ```

5. Get the data from FastF1
    ```
    (.venv) $ python3 download_data.py
    ```
    WARNING: This process may take a while and could quit halfway through. This is due to the fact that FastF1 enforces an API rate limit. I have included some sleeps in the script to avoid this, but it could still happen. If it does, just wait a bit and then restart the script

6. Process the data
    ```
    (.venv) $ python3 processor.py
    ```

7. Train the models
    ```
    (.venv) $ python3 comparison.py
    ```
    Note: Make sure you have a directory called ```models``` in the root of this project.

