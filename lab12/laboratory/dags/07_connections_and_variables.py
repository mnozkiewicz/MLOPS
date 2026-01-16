import datetime
from airflow.sdk import dag, task
from airflow.models import Variable
from airflow.providers.postgres.hooks.postgres import PostgresHook

@dag(
    schedule=datetime.timedelta(minutes=1),
)
def connections_and_variables():

    @task.virtualenv(
        requirements=["twelvedata", "python-dotenv", "pendulum", "lazy-object-proxy"],
        system_site_packages=False,
    )
    def get_data(api_key: str, date_str: str) -> dict:
        from twelvedata import TDClient
        
        td = TDClient(apikey=api_key)
        
        print(f"Fetching data for: {date_str}")

        ts = td.exchange_rate(symbol="USD/EUR", date=date_str)
        return ts.as_json()

    @task()
    def save_data_to_postgress(data: dict):
        if not data:
            raise ValueError("No data")
            
        rate = data.get("rate")
        symbol = f"{data.get('currency_base')}/{data.get('currency_quote')}"
        pg_hook = PostgresHook(postgres_conn_id="POSTGRES_CONN")
        
        pg_hook.run(
            "INSERT INTO exchange_rates (symbol, rate) VALUES (%s, %s)", 
            parameters=(symbol, rate)
        )

    api_key = Variable.get("twelvedata_api_key", default_var="NO_KEY_FOUND")
    data = get_data(api_key=api_key, date_str="{{ ds }}")
    save_data_to_postgress(data)


connections_and_variables()