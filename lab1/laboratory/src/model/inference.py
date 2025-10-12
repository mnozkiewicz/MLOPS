from sklearn.linear_model import LinearRegression
import pandas as pd
from typing import Union, List
import joblib
from pathlib import Path


def load_model() -> LinearRegression:
    current_dir = Path(__file__).parent.resolve()
    model_path = f"{current_dir}/saved_models/linear_model.joblib"
    model: LinearRegression = joblib.load(model_path)

    return model


def predict(
    model: LinearRegression, X: Union[pd.DataFrame, list[list[float]]]
) -> List[str]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    preds = model.predict(X).tolist()
    string_preds = list(map(str, preds))
    return string_preds
