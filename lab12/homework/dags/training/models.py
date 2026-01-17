from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import numpy as np


def create_model(model_type: str, **kwargs):

    match model_type:
        case "ridge":
            return Ridge(**kwargs)
        case "random_forest":
            return RandomForestRegressor(**kwargs)
        case "svm":
            return SVR(**kwargs)
        case _:
            allowed = ["ridge", "rf", "svm"]
            raise ValueError(f"Unknown model type: '{model_type}'. Allowed types: {allowed}")
        

def calculate_metrics(y_true, y_pred) -> dict[str, float]:

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)

    mape = mean_absolute_percentage_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    return {"mae": mae, "rmse": rmse, "mape": mape, "r2": r2}