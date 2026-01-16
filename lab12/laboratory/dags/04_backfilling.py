import openmeteo_requests
import pandas as pd
import pendulum
import datetime
from dotenv import load_dotenv
import os
from airflow.sdk import dag, task
from openmeteo_sdk.WeatherApiResponse import WeatherApiResponse

load_dotenv()


@dag(
    start_date=pendulum.datetime(year=2025, month=1, day=1, tz="America/New_York"),
    end_date=pendulum.datetime(year=2025, month=1, day=31, tz="America/New_York"),
    schedule=datetime.timedelta(weeks=1),
    catchup=True
)
def new_york_data_gathering():

    @task()
    def fetch_weekly_data(**kwargs) -> WeatherApiResponse:
        logical_date: pendulum.DateTime = kwargs["logical_date"]

        start_date = logical_date.add(days=1).to_date_string()
        end_date = logical_date.add(days=7).to_date_string()

        openmeteo = openmeteo_requests.Client()

        url = os.environ["OPENMETEO_API_URL"]
        params = {
            "latitude": 40.7143, 
            "longitude": -74.006,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ["temperature_2m_max", "temperature_2m_min"],
            "timezone": "America/New_York",
        }

        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        daily = response.Daily()

        df = pd.DataFrame(
            {
                "date": pd.to_datetime(daily.Time(), unit="s"),
                "temp_max": daily.Variables(0).ValuesAsNumpy(),
                "temp_min": daily.Variables(1).ValuesAsNumpy(),
            }
        )
        os.makedirs("data", exist_ok=True)
        df.to_csv(f"data/new_york_{start_date}, {end_date}.csv", index=False)
    
    fetch_weekly_data()
    

new_york_data_gathering()