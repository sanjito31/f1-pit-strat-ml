import pandas as pd
from pathlib import Path
import os
       
class F1RaceProcessor:

    def __init__(self, circuit, year, data_dir):
        self.circuit = circuit
        self.year = year
        self.data_dir = data_dir
        self.race = {}


    def load_data(self):

        try: 
            self.race["laps"] = pd.read_parquet(f"{self.data_dir}/{self.year}/{self.circuit}_laps.parquet")
            self.race["weather"] = pd.read_parquet(f"{self.data_dir}/{self.year}/{self.circuit}_weather.parquet")
            self.race["results"] = pd.read_parquet(f"{self.data_dir}/{self.year}/{self.circuit}_results.parquet")
        
        except FileNotFoundError as e:
            return False
        
        return True



    def clean_data(self):

        # Remove NaN lap times
        laps = self.race["laps"].dropna(subset=["LapTime"]).copy()

        # Convert Lap Time to Seconds
        if laps["LapTime"].dtype == "timedelta64[ns]":
            laps["LapTime(s)"] = laps["LapTime"].dt.total_seconds()
        else:
            laps["LapTime(s)"] = laps["LapTime"]

        # Invalid lap times
        laps = laps[(laps["LapTime(s)"] >= 60) & (laps["LapTime(s)"] <= 240)]
        laps['IsValidLap'] = True

        # Tire Compounds
        laps["Compound"] = laps.groupby(["Driver", "Stint"])["Compound"].ffill()
        laps["Compound"] = laps.groupby(["Driver", "Stint"])["Compound"].bfill()

        #Position
        if 'Position' in laps.columns:
            laps['Position'] = laps.groupby('Driver')['Position'].ffill()

        #PersonalBest
        laps['IsPersonalBest'] = laps['IsPersonalBest'].astype('boolean').fillna(False)

        self.race["processed"] = laps


    def dummy_encoding(self):

        laps = self.race["processed"]

        # Encode tire compound
        categories = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]
        laps["Compound"] = pd.Categorical(laps["Compound"], categories=categories)
        dummies = pd.get_dummies(laps["Compound"], prefix="Tire")
        laps = pd.concat([laps, dummies], axis=1)

        # Encode race phase
        race_phase = pd.cut(laps["LapNumber"], bins=3, 
                                labels=["Early", "Middle", "Late"])
        laps = pd.concat([laps, pd.get_dummies(race_phase, prefix="RacePhase")], axis=1)

        # Encode Team name
        teams = pd.get_dummies(laps["Team"], prefix="Team")
        laps = pd.concat([laps, teams], axis=1)

        # Encode Driver Name
        drivers = pd.get_dummies(laps["Driver"], prefix="Driver")
        laps = pd.concat([laps, drivers], axis=1)

        # Encode Track Status
        track_status_codes = {
            'status_clear': '1',
            'status_yellow_flag': '2', 
            'status_safety_car': '4',
            'status_red_flag': '5',
            'status_vsc': '6',
            'status_vsc_ending': '7'
        }

        for status in track_status_codes.keys():
            laps[status] = 0

        for status, code in track_status_codes.items():
            laps[status] = laps["TrackStatus"].str.contains(code, na=False).astype(int)

        self.race["processed"] = laps


    def engineer_pace_features(self):

        laps = self.race["processed"].copy()

        # Pace drop off per lap per driver
        laps = laps.sort_values(["LapNumber"])
        laps["PaceDropoff"] = laps.groupby("Driver")["LapTime(s)"].transform(
            lambda x: x - x.expanding().min()
        ).round(3)

        # Last 3 lap time moving average
        laps = laps.sort_values(["LapNumber"])
        laps["LapTimeMA3"] = laps.groupby("Driver")["LapTime(s)"].transform(
            lambda x: x.rolling(window=3).mean().fillna(x)
        ).round(3)

        # Lap time trend based on last 3 lap moving average
        laps = laps.sort_values(["LapNumber"])
        laps["LapTimeTrend"] = laps.groupby("Driver")["LapTimeMA3"].transform(
            lambda x: x.diff().fillna(0)
        ).round(3)

        # Position changes per lap per driver
        laps = laps.sort_values(["LapNumber"])
        laps["PositionChange"] = laps.groupby("Driver")["Position"].transform(
            lambda x: x.diff().fillna(0)
        )

        # Straight line speed drop off
        laps = laps.sort_values(["LapNumber"])
        laps["SpeedDropoff"] = laps.groupby("Driver")["SpeedST"].transform(
            lambda x: x - x.expanding().max()
        ).round(0)

        # Gap to driver ahead per lap
        laps = laps.sort_values(["Position"])
        laps["GapToAhead(ms)"] = laps.groupby("LapNumber")["Time"].transform(
            lambda x: (x.diff().fillna(pd.Timedelta(0))).dt.total_seconds() * 1000
        ).round(3)

        # Should Pit Next determination
        laps["ShouldPitNext"] = laps["PitInTime"].shift(-1).notna()

        self.race["processed"] = laps


    def add_weather(self):

        laps = self.race["processed"].copy()

        laps = pd.merge_asof(laps.sort_values("Time"),
                                self.race["weather"].sort_values("Time"),
                                on="Time",
                                direction="nearest")
        
        self.race["processed"] = laps


    def drop_unused_columns(self):

        self.race["processed"] = self.race["processed"].drop([
                        "Time",
                        "Driver", 
                        "DriverNumber", 
                        "LapTime", 
                        "Compound", 
                        "LapNumber",
                        "PitOutTime",
                        "PitInTime",
                        "Sector1Time",
                        "Sector2Time",
                        "Sector3Time",
                        "Sector1SessionTime",
                        "Sector2SessionTime",
                        "Sector3SessionTime",
                        "SpeedI1",
                        "SpeedI2",
                        "SpeedFL",
                        "SpeedST",
                        "IsPersonalBest",
                        "Team",
                        "LapStartTime",
                        "LapStartDate",
                        "TrackStatus",
                        "Deleted",
                        "DeletedReason",
                        "FastF1Generated",
                        "IsAccurate"], axis=1)
        
    def save_to_parquet(self, output):

        dir = os.path.join(f"{output}", f"{self.year}")
        Path(dir).mkdir(parents=True, exist_ok=True)

        self.race["processed"].to_parquet(f'{dir}/{self.circuit}_processed.parquet')


def main():

    input = "data/raw"
    output = "data/processed"

    # years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    years = [2024]
    # tracks = ["abudhabi", "bahrain", "monaco", "silverstone", "singapore"]
    tracks = ["sao_paulo"]

    for year in years:
        for track in tracks:

            ## Create race object
            race = F1RaceProcessor(track, year, input)

            ## Load data
            if race.load_data() is False:
                print(f"Could not load data from {track}, {year}, {input}")
                continue

            ## Clean data
            race.clean_data()

            ## Process data
            race.dummy_encoding()
            race.engineer_pace_features()
            race.add_weather()
            race.drop_unused_columns()

            ## Save data to file
            race.save_to_parquet(output)

if __name__ == "__main__":
    main()


