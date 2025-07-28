import pandas as pd
from pathlib import Path
import json
       
class F1RaceProcessor:

    def __init__(self, circuit, year, data_dir):
        self.circuit = circuit
        self.year = year
        self.data_dir = data_dir
        self.race = {}


    def load_data(self):

        try: 
            base_path = Path(self.data_dir) / str(self.year)
            self.race["laps"] = pd.read_parquet(base_path / f"{self.circuit}_laps.parquet")
            self.race["weather"] = pd.read_parquet(base_path / f"{self.circuit}_weather.parquet")
            self.race["results"] = pd.read_parquet(base_path / f"{self.circuit}_results.parquet")
        
        except FileNotFoundError as e:
            return False
        
        return True



    def clean_laps_data(self):

        # Remove NaN lap times
        laps = self.race["laps"].dropna(subset=["LapTime"]).copy()

        # Convert Lap Time to Seconds
        if laps["LapTime"].dtype == "timedelta64[ns]":
            laps["LapTime(s)"] = laps["LapTime"].dt.total_seconds().round(3)
        else:
            laps["LapTime(s)"] = laps["LapTime"]

        # Invalid lap times
        laps = laps[(laps["LapTime(s)"] >= 60) & (laps["LapTime(s)"] <= 240)]
        laps['IsValidLap'] = True

        # Lap Number
        laps = laps.sort_values(["Time"])
        laps["LapNumber"] = laps.groupby("Driver")["LapNumber"].transform(
            lambda group: group.interpolate(method='linear').round()
        )

        # Tire Compounds
        laps["Compound"] = laps.groupby(["Driver", "Stint"])["Compound"].ffill()
        laps["Compound"] = laps.groupby(["Driver", "Stint"])["Compound"].bfill()

        # Position
        laps = laps.sort_values(['LapNumber', 'Time'])
        laps['Position'] = laps.groupby('LapNumber')["Position"].transform(
            lambda group: group.astype(float).fillna(laps.loc[group.index, 'Time'].rank(method='dense')).round()
        )

        # TyreLife
        laps = laps.sort_values(["LapNumber"])
        laps['TyreLife'] = laps.groupby(["Driver", "Stint"])["TyreLife"].transform(
            lambda group: range(1, len(group) + 1)
        )

        # SpeedST (Speed on the longest straight)
        laps = laps.sort_values(["LapNumber"])
        laps["SpeedST"] = laps.groupby(["Driver"])["SpeedST"].ffill()
        laps["SpeedST"] = laps.groupby(["Driver"])["SpeedST"].bfill()

        # PersonalBest
        laps['IsPersonalBest'] = laps['IsPersonalBest'].astype('boolean').fillna(False)

        self.race["processed"] = laps


    def clean_weather_data(self):

        weather = self.race["weather"].copy()

        # Use linear interpolation for weather data since weather data is generally continuous 
        weather_cols = ['AirTemp', 'Humidity', 'Pressure', 'TrackTemp', 'WindSpeed', 'WindDirection']
        weather[weather_cols] = weather[weather_cols].interpolate(method="linear", limit=2)
        weather['WindDirection'] = weather['WindDirection'] % 360
        weather['Rainfall'] = weather['Rainfall'].astype('boolean').ffill().fillna(0)

        self.race["weather"] = weather


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
        laps["GapToAhead(s)"] = laps.groupby("LapNumber")["Time"].transform(
            lambda x: (x.diff().fillna(pd.Timedelta(0))).dt.total_seconds()
        ).round(3)

        # Should Pit Next determination
        laps = laps.sort_values(["Driver", "LapNumber"])
        laps["pit_occurred"] = (laps.groupby("Driver")['Stint'].diff() != 0).shift(-1).astype('boolean').fillna(False)
        laps["ShouldPitNext"] = laps.groupby("Driver")["pit_occurred"].shift(-1).astype('boolean').fillna(False)
        laps = laps.drop("pit_occurred", axis=1)

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

        output_dir = Path(output) / str(self.year)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.race["processed"].to_parquet(output_dir / f'{self.circuit}_processed.parquet')


def main():

    input = Path("data/raw")
    output = Path("data/processed")

    # NO 2018 SINCE MANY GAPS IN STINT AND TYRELIFE DATA
    # 2019 - 2023 USED FOR TRAINING
    # 2024 USED FOR VALIDATION

    data_start_year = 2019
    data_end_year = 2023

    for year in range(data_start_year, data_end_year + 1):

        tracks = {}
        with open(input / f"{year}" / "schedule.json", "r") as f:
            tracks = json.load(f)

        for _, event in tracks.items():

            track = event["name"]

            ## Create race object
            race = F1RaceProcessor(track, year, input)

            ## Load data
            if race.load_data() is False:
                print(f"Could not load data from {track}, {year}, {input}")
                continue

            ## Clean data
            race.clean_laps_data()
            race.clean_weather_data()

            ## Process data
            race.dummy_encoding()
            race.engineer_pace_features()
            race.add_weather()
            race.drop_unused_columns()

            ## NA Report
            missing = race.race["processed"].isna().sum().sum()
            if missing > 0:
                print(f"{track}, {year} has {missing} values missing.")

            ## Save data to file
            race.save_to_parquet(output)

if __name__ == "__main__":
    main()


