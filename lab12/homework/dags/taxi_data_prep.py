import pendulum
import requests
from airflow.sdk import dag, task, ObjectStoragePath

@dag(
    start_date=pendulum.datetime(year=2024, month=1, day=1, tz="America/New_York"),
    end_date=pendulum.datetime(year=2024, month=12, day=1, tz="America/New_York"),
    schedule="@monthly",
    catchup=True,
    max_active_runs=1,
)
def taxi_data_prep():

    @task()
    def download_monthly_data(**kwargs) -> tuple[int, int]:

        logical_date = kwargs["logical_date"]
        year = logical_date.year
        month = logical_date.month

        filename = f"yellow_tripdata_{year}-{month:02}.parquet"
        url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{filename}"
        s3_path = ObjectStoragePath(f"s3://aws_default@datasets/taxi_data/{filename}")

        if s3_path.exists():
            print(f"File {filename} already exists. Skipping.")
            return

        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            with s3_path.open("wb") as s3_file:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        s3_file.write(chunk)
        
        print(f"Uploaded {filename}")

        return year, month

    @task.virtualenv(
        requirements=["polars", "s3fs", "fsspec", "pendulum", "lazy-object-proxy"],
        system_site_packages=False,
    )
    def process_data_polars(date_args: list[int]):
        import sys
        sys.path.append("/opt/airflow/dags")

        from processing.process_data import process_data


        year, month = date_args
        if year == -1:
            print("Previous task indicated no data. Skipping processing.")
            return

        print(f"Processing data for {year}-{month:02}...")

        df_processed = process_data(year, month, pipeline_run=True)
        output_path = f"s3://datasets/processed_data/year={year}/month={month}/data.parquet"
        storage_options = {
            "endpoint_url": "http://localstack:4566",
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
            "aws_region": "us-east-1",
            "allow_http": "true"
        }
        print(f"Saving processed data to {output_path}")
        df_processed.write_parquet(output_path, storage_options=storage_options)


    year_and_month = download_monthly_data()
    process_data_polars(year_and_month)


taxi_data_prep()