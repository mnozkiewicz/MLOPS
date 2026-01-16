import datetime
from airflow.sdk import dag, task

@dag(
    schedule=datetime.timedelta(minutes=1),
)
def scheduling_dataset_gathering():

    @task.virtualenv(
        requirements=["twelvedata", "python-dotenv", "pendulum", "lazy-object-proxy"],
        system_site_packages=False,
    )
    def get_data(logical_date) -> dict:
        import os
        from dotenv import load_dotenv
        from twelvedata import TDClient

        load_dotenv()

        td = TDClient(apikey=os.environ["TWELVEDATA_API_KEY"])
        ts = td.exchange_rate(symbol="USD/EUR", date=logical_date.isoformat())
        data = ts.as_json()

        return data

    @task.virtualenv(
        requirements=["pendulum", "lazy-object-proxy"],
        system_site_packages=False,
    )
    def save_data(data: dict) -> None:
        import json
        import os

        if not data:
            raise ValueError("No data received")

        os.makedirs("data", exist_ok=True)

        with open("data/data.jsonl", "a+") as file:
            file.write(json.dumps(data))
            file.write("\n")

    data = get_data()
    save_data(data)


scheduling_dataset_gathering()
