import pandas as pd
import requests
import pendulum
from airflow.decorators import dag, task
from airflow.sdk import ObjectStoragePath


@dag()
def weather_etl_s3():

    @task()
    def get_data() -> dict:
        print("Fetching data from API")
        url = "https://archive-api.open-meteo.com/v1/archive?latitude=40.7143&longitude=-74.006&start_date=2025-01-01&end_date=2025-12-31&hourly=temperature_2m&timezone=auto"

        resp = requests.get(url)
        resp.raise_for_status()

        data = resp.json()
        data = {
            "time": data["hourly"]["time"],
            "temperature": data["hourly"]["temperature_2m"],
        }
        return data

    @task()
    def transform(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data)
        df["temperature"] = df["temperature"].clip(lower=-20, upper=50)
        return df

    @task()
    def save_data(df: pd.DataFrame, **kwargs) -> None:
        print("Saving the data to S3")

        date_str = kwargs['ds']
        filename = f"weather_{date_str}.csv"
        s3_path = ObjectStoragePath(f"s3://aws_default@weather-data/{filename}")

        with s3_path.open("w") as f:
            df.to_csv(f, index=False)
            
        print(f"Successfully saved to {s3_path}")

    data = get_data()
    transformed = transform(data)
    save_data(transformed)


weather_etl_s3()