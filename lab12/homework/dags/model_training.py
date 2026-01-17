from airflow.sdk import dag, task
from airflow.providers.postgres.hooks.postgres import PostgresHook
import json

@dag()
def model_training_pipeline():

    @task.virtualenv(
        requirements=["polars", "s3fs", "fsspec"],
        system_site_packages=False
    )
    def prepare_dataset() -> dict:
        import polars as pl
        
        storage_options = {
            "endpoint_url": "http://localstack:4566",
            "aws_access_key_id": "test",
            "aws_secret_access_key": "test",
            "aws_region": "us-east-1",
            "allow_http": "true"
        }

        year = 2024
        source_prefix = "s3://datasets/processed_data"
        staging_prefix = f"s3://datasets/ml_staging/{year}"

        training_months = list(range(1, 12))
        train_dfs = []

        for month in training_months:
            path = f"{source_prefix}/year={year}/month={month}/data.parquet"
            df = pl.scan_parquet(path, storage_options=storage_options)
            train_dfs.append(df)

        train_df = pl.concat(train_dfs)

        train_output_path = f"{staging_prefix}/train.parquet"
        train_df.sink_parquet(train_output_path, storage_options=storage_options)


        test_path_source = f"{source_prefix}/year={year}/month=12/data.parquet"
        test_output_path = f"{staging_prefix}/test.parquet"

        test_df = pl.scan_parquet(test_path_source, storage_options=storage_options)
        test_df.sink_parquet(test_output_path, storage_options=storage_options)

        return {
            "train_path": train_output_path,
            "test_path": test_output_path
        }
    

    @task.virtualenv(
        task_id="train_model",
        requirements=["pandas", "scikit-learn", "s3fs", "pyarrow"],
        system_site_packages=False
    )
    def train_model(paths: dict, model_type: str, hyperparams: dict):
        import sys
        import pickle
        import s3fs
        import numpy as np
        
        sys.path.append("/opt/airflow/dags")

        from training.data_transform import load_dataset, create_preprocessor
        from training.models import create_model, calculate_metrics

        df_train = load_dataset(paths['train_path'], pipeline_run=True)
        df_test = load_dataset(paths['test_path'], pipeline_run=True)

        y_train = np.log1p(df_train.pop("total_ride_count").to_numpy())
        y_test = df_test.pop("total_ride_count").to_numpy()

        preprocessor = create_preprocessor()
        X_train = preprocessor.fit_transform(df_train)
        X_test = preprocessor.transform(df_test)


        model = create_model(model_type, **hyperparams)
        
        model.fit(X_train, y_train)
        y_predict_log = model.predict(X_test)
        y_predict = np.expm1(y_predict_log)
        
        metrics = calculate_metrics(y_test, y_predict)

        s3_config = {
            "endpoint_url": "http://localstack:4566",
            "key": "test",
            "secret": "test",
        }
        fs = s3fs.S3FileSystem(client_kwargs={"endpoint_url": s3_config["endpoint_url"]}, 
                               key=s3_config["key"], 
                               secret=s3_config["secret"])
        
        model_filename = f"model_{model_type}.pkl"
        s3_model_path = f"s3://datasets/models/{model_filename}"
        
        print(f"Saving trained model to {s3_model_path}...")
        final_pipeline = {
            "preprocessor": preprocessor,
            "model": model
        }
        with fs.open(s3_model_path, "wb") as f:
            pickle.dump(final_pipeline, f)

        return {
            "model_type": model_type,
            "mae": metrics["mae"],
            "mape": metrics["mape"],
            "rmse": metrics["rmse"],
            "r2": metrics["r2"],
            "model_path": s3_model_path,
            "hyperparams": hyperparams,
        }
    

    @task()
    def save_results_to_postgres(results: list):
        if not results:
            return

        rows_to_insert = []
        for res in results:
            row = (
                res.get("model_type"),
                json.dumps(res.get("hyperparams", {})),
                res.get("mae"),
                res.get("mape"),
                res.get("rmse"),
                res.get("r2"),
                res.get("model_path")
            )
            rows_to_insert.append(row)

        pg_hook = PostgresHook(postgres_conn_id="POSTGRES_CONN")
        pg_hook.insert_rows(
            table="model_metrics",
            rows=rows_to_insert,
            target_fields=["model_type", "hyperparameters", "mae", "mape", "rmse", "r2", "model_path"]
        )
    
    data_paths = prepare_dataset()

    model_configs = [
        # Ridge Regression
        {"model_type": "ridge", "hyperparams": {"alpha": 0.1}},
        {"model_type": "ridge", "hyperparams": {"alpha": 1.0}},
        {"model_type": "ridge", "hyperparams": {"alpha": 10.0}},
        # Random Forest
        {"model_type": "random_forest", "hyperparams": {"n_estimators": 50, "max_depth": 5}},
        {"model_type": "random_forest", "hyperparams": {"n_estimators": 100, "max_depth": 10}},
        {"model_type": "random_forest", "hyperparams": {"n_estimators": 200, "max_depth": 15}},
        # SVM
        {"model_type": "svm", "hyperparams": {"C": 0.1, "kernel": "rbf"}},
        {"model_type": "svm", "hyperparams": {"C": 1.0, "kernel": "rbf"}},
        {"model_type": "svm", "hyperparams": {"C": 10.0, "kernel": "rbf"}},
    ]

    training_results = train_model.partial(paths=data_paths).expand_kwargs(model_configs)
    save_results_to_postgres(training_results)

model_training_pipeline()