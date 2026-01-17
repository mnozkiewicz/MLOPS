import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def load_dataset(s3_path: str, pipeline_run: bool = False) -> pd.DataFrame:
    host = "localstack" if pipeline_run else "localhost"
    storage_options = {
        "client_kwargs": {"endpoint_url": f"http://{host}:4566"},
        "key": "test",
        "secret": "test",
    }

    df = pd.read_parquet(s3_path, storage_options=storage_options)
    return df


def create_preprocessor() -> ColumnTransformer:
    numeric_selector = make_column_selector(dtype_include=['int64', 'float64', 'uint32'])
    numeric_transformer = StandardScaler()

    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_selector = make_column_selector(dtype_include=['int8', 'bool'])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_selector),
            ("cat", categorical_transformer, categorical_selector),
        ],
        remainder="drop"  
    )
    return preprocessor
