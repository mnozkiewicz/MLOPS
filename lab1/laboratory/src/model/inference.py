from sklearn.linear_model import LogisticRegression
import pandas as pd
from typing import Union, List
import joblib
from pathlib import Path


def load_model() -> LogisticRegression:
    current_dir = Path(__file__).parent.resolve()
    model_path = f"{current_dir}/saved_models/model.joblib"
    model: LogisticRegression = joblib.load(model_path)

    return model


def predict(
    model: LogisticRegression, X: Union[pd.DataFrame, list[list[float]]]
) -> List[str]:
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    preds = model.predict(X).tolist()
    string_preds = list(map(str, preds))
    return string_preds
