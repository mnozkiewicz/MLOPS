from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
from typing import Tuple
import pandas as pd
import joblib
from pathlib import Path


def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    df, y = load_iris(return_X_y=True, as_frame=True)
    df.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
    return df, y


def train_model(df: pd.DataFrame, y: pd.Series) -> LogisticRegression:
    model = LogisticRegression(max_iter=100)
    model.fit(df, y)
    return model


def save_model(model: LogisticRegression) -> None:
    current_dir = Path(__file__).parent.resolve()
    saving_path = f"{current_dir}/saved_models/model.joblib"
    joblib.dump(model, saving_path)


if __name__ == "__main__":
    df, y = load_data()
    model = train_model(df, y)
    save_model(model)
