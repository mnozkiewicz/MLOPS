import polars as pl
from datetime import time

def load_data_from_s3(year: int, month: int, bucket: str = "datasets", pipeline_run: bool = False) -> pl.LazyFrame:

    host = "localstack" if pipeline_run else "localhost"
    storage_options = {
        "endpoint_url": f"http://{host}:4566",
        "aws_access_key_id": "test",
        "aws_secret_access_key": "test",
        "aws_region": "us-east-1",
        "allow_http": "true"
    }
    
    filename = f"yellow_tripdata_{year}-{month:02}.parquet"
    s3_path = f"s3://{bucket}/taxi_data/{filename}"
    
    print(f"Loading data from: {s3_path}")
    return pl.scan_parquet(
        s3_path,
        storage_options=storage_options
    )

def cast_and_filter_columns(df: pl.LazyFrame, year: int, month: int) -> pl.LazyFrame:
    return (
        df
        .filter(
            pl.col("tpep_pickup_datetime").dt.year() == year,
            pl.col("tpep_dropoff_datetime").dt.month() == month,
        )
        .with_columns(
            pl.col("VendorID").cast(pl.UInt8),
            pl.col("passenger_count").cast(pl.UInt8),
            pl.col("store_and_fwd_flag").cast(pl.Categorical),
            pl.col("RatecodeID").cast(pl.UInt8),
            pl.col("PULocationID").cast(pl.UInt16),
            pl.col("DOLocationID").cast(pl.UInt16),
            pl.col("payment_type").cast(pl.UInt8)
        )
    )


def clean_data(df: pl.LazyFrame) -> pl.LazyFrame:

    money_related_cols = [
        "fare_amount", "extra", "mta_tax", "tip_amount", "tolls_amount",
        "improvement_surcharge", "total_amount", "congestion_surcharge", "Airport_fee"
    ]
    valid_ratecode_ids = [1, 2, 3, 4, 5, 6, 99]
    valid_vendor_ids = [1, 2, 6, 7]

    return (
        df
        .with_columns(
            pl.col("passenger_count").fill_null(1)
        )
        .filter(pl.col("passenger_count") > 0)
        .with_columns(
            pl.when(pl.col("passenger_count") > 6)
            .then(pl.lit(6))
            .otherwise(pl.col("passenger_count"))
            .alias("passenger_count")
        )
        .with_columns(
            (pl.col("tpep_dropoff_datetime") - pl.col("tpep_pickup_datetime"))
            .dt.total_minutes()
            .alias("trip_duration_minutes")
        )
        .filter(
            (pl.col("trip_duration_minutes") > 0) & 
            (pl.col("trip_duration_minutes") <= 120)
        )
        .with_columns([pl.col(c).abs() for c in money_related_cols])
        .filter(
            pl.all_horizontal([pl.col(c) <= 1000 for c in money_related_cols])
        )
        .filter(
            pl.col("RatecodeID").is_in(valid_ratecode_ids),
            pl.col("VendorID").is_in(valid_vendor_ids)
        )
    )

def add_trip_features(df: pl.LazyFrame) -> pl.LazyFrame:
    is_weekday = pl.col("tpep_pickup_datetime").dt.weekday().is_between(1, 5)
    is_morning_rush = (
        (pl.col("tpep_pickup_datetime").dt.time() >= time(6, 30)) & 
        (pl.col("tpep_pickup_datetime").dt.time() < time(9, 30))
    )
    is_evening_rush = (
        (pl.col("tpep_pickup_datetime").dt.time() >= time(15, 30)) &
        (pl.col("tpep_pickup_datetime").dt.time() < time(20, 0))
    )

    rush_hour_expr = is_weekday & (is_morning_rush | is_evening_rush)

    return (
        df
        .with_columns(
            pl.when(pl.col("payment_type") == 1).then(pl.lit("card"))
            .when(pl.col("payment_type") == 2).then(pl.lit("cash"))
            .otherwise(pl.lit("other"))
            .alias("payment_type_desc")
            .cast(pl.Categorical),
            (pl.col("Airport_fee") > 0).alias("is_airport_ride"),
            rush_hour_expr.alias("is_rush_hour")
        )
    )

def aggregate_daily_stats(df: pl.LazyFrame) -> pl.DataFrame:

    group_cols = [pl.col("tpep_pickup_datetime").dt.date().alias("date")]
    return (
        df
        .collect()
        .to_dummies(["payment_type_desc"]) 
        .group_by(group_cols)
        .agg(
            pl.col("^payment_type_desc_.*$").sum(),
            pl.col("is_airport_ride").sum().alias("total_airport_rides"),
            pl.col("is_rush_hour").sum().alias("total_rush_hour_rides"),
            pl.col("passenger_count").sum().alias("total_passengers"),
            pl.col("fare_amount").mean().alias("avg_fare_amount"),
            pl.col("trip_distance").median().alias("median_trip_distance"),
            pl.col("total_amount").sum().alias("total_revenue"),
            pl.col("congestion_surcharge").sum().alias("total_congestion_surcharge"),
            pl.len().alias("total_ride_count")
        )
        .with_columns(
            pl.col("date").dt.quarter().alias("quarter"),
            pl.col("date").dt.month().alias("month"),
            pl.col("date").dt.day().alias("day_of_month"),
            pl.col("date").dt.weekday().alias("day_of_week"),
            pl.col("date").dt.weekday().is_between(6, 7).alias("is_weekend")
        )
        .sort("date")
    )

def process_data(year: int, month: int, pipeline_run: bool = False) -> pl.DataFrame:
    
    df_raw = load_data_from_s3(year, month, pipeline_run=pipeline_run)
    df_clean = cast_and_filter_columns(df_raw, year, month)
    df_clean = clean_data(df_clean)
    df_enriched = add_trip_features(df_clean)
    df_agg = aggregate_daily_stats(df_enriched)
    return df_agg