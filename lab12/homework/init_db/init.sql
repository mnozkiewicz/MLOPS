CREATE TABLE IF NOT EXISTS model_metrics (
    id SERIAL PRIMARY KEY,
    training_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    model_type VARCHAR(50) NOT NULL,
    hyperparameters JSONB,
    mae FLOAT,
    mape FLOAT,
    rmse FLOAT,
    r2 FLOAT,
    model_path VARCHAR(255)
);