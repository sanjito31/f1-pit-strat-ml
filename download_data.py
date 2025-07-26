import fastf1
from pathlib import Path


def save_race_data(year, round_num, circuit_name):

    Path(f"data/raw/{year}").mkdir(parents=True, exist_ok=True)

    race = fastf1.get_session(year, round_num, 'R')
    race.load()

    base_path = f"data/raw/{year}/{circuit_name}"

    race.laps.to_parquet(f"{base_path}_laps.parquet")
    race.results.to_parquet(f"{base_path}_results.parquet")
    race.weather_data.to_parquet(f"{base_path}_weather.parquet")

    print(f"Saved {circuit_name} {year} data")


def main():

    circuits = [
        (2024, 1, 'bahrain'),
        (2024, 8, 'monaco'),
        (2024, 12, 'silverstone'),
        (2024, 18, 'singapore'),
        (2024, 24, 'abudhabi')
    ]

    for year, round_num, name in circuits:
        save_race_data(year, round_num, name)


if __name__ == "__main__":
    main()