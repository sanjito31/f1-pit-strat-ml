import fastf1
from pathlib import Path
import time
import os
import json


def save_race_data(year, round_num, circuit_name):

    try:

        Path(f"data/raw/{year}").mkdir(parents=True, exist_ok=True)

        base_path = f"data/raw/{year}/{circuit_name}"

        if(os.path.exists(f"{base_path}_laps.parquet")):
            print(f"{circuit_name}, {year} already downloaded. Skipping")
            return

        race = fastf1.get_session(year, round_num, 'R')
        race.load()

        race.laps.to_parquet(f"{base_path}_laps.parquet")
        race.results.to_parquet(f"{base_path}_results.parquet")
        race.weather_data.to_parquet(f"{base_path}_weather.parquet")

        print(f"Saved {circuit_name} {year} data")
    except Exception as e:
        print(f"Failed: {year}, {circuit_name}. ({e})")


def season_to_dict(year):

    schedule = {}
    season = fastf1.get_event_schedule(year)

    for _, event in season.iterrows():
        
        round_num = event["RoundNumber"]
        if round_num != 0:
            schedule[round_num] = {
                "name": event["EventName"].replace(" Grand Prix", "").lower().replace(" ", "_"),
                "country": event["Country"],
                "location": event["Location"],
                "race_start_utc": event["Session5DateUtc"].isoformat()
            }

    return schedule

def main():

    raw_path = 'data/raw'
    years = [2018, 2019, 2020, 2021, 2022, 2023, 2024]
    

    for year in years:

        try:

            season = season_to_dict(year)

            for num, details in season.items():

                race_name = details.get("name")

                # Pre-season testing
                if num < 1 or "test" in race_name:
                    continue

                # for Sao Paulo
                if "ã" in race_name:
                    race_name = race_name.replace("ã", "a")

                file_path = os.path.join(raw_path, f"{year}", f"{race_name}_laps.parquet")

                # Skip saving race data if already present
                if os.path.exists(file_path):
                    print(f"{race_name}, {year} already downloaded. Skipping...")
                    continue
                else:
                    save_race_data(year, num, race_name)
                    time.sleep(2)

            # Finally, save season info to json
            json_path = os.path.join(raw_path, f"{year}", "schedule.json")
            with open(json_path, "w") as f:
                json.dump(season, f, indent=2)
            
            time.sleep(5)

        except Exception as e:
            print(e)
            continue




if __name__ == "__main__":
    main()