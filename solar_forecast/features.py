"""
Construct Physics-Informed Features 
"""
import numpy as np
import pandas as pd
from pathlib import Path
from loguru import logger
import typer

from solar_forecast.config import (
    MERGED_OUT,
    MODEL_READY_OUT,
    FIGURES_ROOT,
    TS_COL as ts_col,
    ZONE_COL as zone_col,
    TARGET as target,
)

app = typer.Typer()

FINAL_FEATURES = [
    "forecast_mw",
    "temperature_2m",
    "surface_pressure",
    "cloud_cover",
    "windspeed_10m",
    "shortwave_radiation",
    "hour_sin",
    "hour_cos",
    "month_sin",
    "month_cos",
    "dayofyear_sin",
    "dayofyear_cos",
    "forecast_x_hour_sin",
    "forecast_x_hour_cos",
    "shortwave_x_cloud",
    "shortwave_x_temp",
    "forecast_roll_mean_3",
    "shortwave_roll_mean_3",
    "forecast_roll_mean_24",
    "shortwave_roll_mean_24",
    "forecast_diff_1",
    "shortwave_diff_1",
    "shortwave_ramp_abs",
    "is_morning_ramp",
    "is_midday",
]


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofyear_local"] = df["time_local"].dt.dayofyear

    df["hour_sin"] = np.sin(2 * np.pi * df["hour_local"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour_local"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month_local"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month_local"] / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear_local"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear_local"] / 365.25)

    return df


def add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_morning_ramp"] = df["hour_local"].between(6, 9).astype(int)
    df["is_midday"] = df["hour_local"].between(10, 14).astype(int)
    return df


def add_interact_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_x_hour_sin"] = df["forecast_mw"] * df["hour_sin"]
    df["forecast_x_hour_cos"] = df["forecast_mw"] * df["hour_cos"]
    df["shortwave_x_cloud"] = df["shortwave_radiation"] * (df["cloud_cover"] / 100.0)
    df["shortwave_x_temp"] = df["shortwave_radiation"] * df["temperature_2m"]
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_roll_mean_3"] = df["forecast_mw"].shift(1).rolling(3, min_periods=1).mean()
    df["shortwave_roll_mean_3"] = df["shortwave_radiation"].shift(1).rolling(3, min_periods=1).mean()
    df["forecast_roll_mean_24"] = df["forecast_mw"].shift(1).rolling(24, min_periods=1).mean()
    df["shortwave_roll_mean_24"] = df["shortwave_radiation"].shift(1).rolling(24, min_periods=1).mean()
    df["forecast_diff_1"] = df["forecast_mw"].diff(1)
    df["shortwave_diff_1"] = df["shortwave_radiation"].diff(1)
    df["shortwave_ramp_abs"] = df["shortwave_diff_1"].abs()
    return df


def load_and_standardize_data() -> pd.DataFrame:
    logger.info(f"Loading data from {MERGED_OUT}.")
    df = pd.read_csv(MERGED_OUT, low_memory=False)

    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    df[ts_col] = pd.to_datetime(df[ts_col], utc=True, errors="coerce")

    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
        same_ts_mask = (df["time"] == df[ts_col]) | (df["time"].isna() & df[ts_col].isna())
        if bool(same_ts_mask.all()):
            df = df.drop(columns=["time"])

    numeric_cols = [
        "actual_mw",
        "forecast_mw",
        "capacity_mw",
        "temperature_2m",
        "surface_pressure",
        "cloud_cover",
        "windspeed_10m",
        "shortwave_radiation",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[zone_col] = df[zone_col].astype(str).str.strip().str.upper()

    logger.info(f"Loaded and standardized data: {df.shape}.")
    return df


def add_time_context(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_local"] = df[ts_col].dt.tz_convert("America/New_York")
    df["date_local"] = df["time_local"].dt.date
    df["year_from_ts"] = df["time_local"].dt.year
    df["month_local"] = df["time_local"].dt.month
    df["dayofweek_local"] = df["time_local"].dt.dayofweek
    df["hour_local"] = df["time_local"].dt.hour

    df["is_weekend"] = df["dayofweek_local"].isin([5, 6]).astype(int)
    df["is_daylight_proxy"] = (df["shortwave_radiation"] > 0).astype(int)

    return df


def add_target_and_errors(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[target] = df["actual_mw"] - df["forecast_mw"]
    df["absolute_error_mw"] = (df["actual_mw"] - df["forecast_mw"]).abs()
    df["smape"] = np.where(
        (df["actual_mw"].abs() + df["forecast_mw"].abs()) > 0,
        200 * df["absolute_error_mw"] / (df["actual_mw"].abs() + df["forecast_mw"].abs()),
        np.nan
    )

    return df


def extract_system_level(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Extracting system-level data.")
    df_system = (
        df[df[zone_col] == "SYSTEM"]
        .copy()
        .sort_values(ts_col)
        .reset_index(drop=True)
    )
    logger.info(f"System-level shape: {df_system.shape}.")
    return df_system


def engineer_features(df_system: pd.DataFrame) -> pd.DataFrame:
    logger.info("Engineering physics-informed features.")

    df_system = add_cyclic_features(df_system)
    df_system = add_regime_flags(df_system)
    df_system = add_interact_features(df_system)
    df_system = add_rolling_features(df_system)

    logger.info(f"Features engineered. Shape: {df_system.shape}.")
    return df_system


def create_train_test_split(df_system: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    logger.info(f"Splitting data at {split_date}.")

    y = df_system[target].copy()
    train_mask = df_system[ts_col] < split_date
    test_mask = df_system[ts_col] >= split_date

    train_idx = train_mask & y.notna()
    test_idx = (
        test_mask
        & y.notna()
        & df_system["actual_mw"].notna()
        & df_system["forecast_mw"].notna()
    )

    df_system["dataset_split"] = np.where(df_system[ts_col] < split_date, "train", "test")

    logger.info(f"Train rows: {train_idx.sum()}, Test rows: {test_idx.sum()}.")
    return df_system


def build_model_ready_dataset(df_system: pd.DataFrame) -> pd.DataFrame:
    logger.info("Building model-ready dataset.")

    y = df_system[target].copy()

    drop_cols = [
        "actual_mw",
        "absolute_error_mw",
        "smape",
        ts_col,
        "time_local",
        "date_local",
        zone_col,
        target,
        "capacity_mw",
        "year_from_ts",
        "month_local",
        "dayofweek_local",
        "hour_local",
        "is_weekend",
    ]

    base_cols = [
        ts_col,
        "time_local",
        zone_col,
        "dataset_split",
        "actual_mw",
        "forecast_mw",
        target,
    ]

    final_features = [c for c in FINAL_FEATURES if c in df_system.columns]

    out_cols = []
    for c in base_cols + final_features:
        if c not in out_cols:
            out_cols.append(c)

    model_ready = df_system.loc[
        y.notna() & df_system["forecast_mw"].notna(),
        out_cols
    ].copy()

    logger.info(f"Model-ready dataset shape: {model_ready.shape}.")
    logger.info(f"Columns: {len(model_ready.columns)}.")

    return model_ready


@app.command()
def main(
    input_path: Path = MERGED_OUT,
    output_path: Path = MODEL_READY_OUT,
):
    logger.info("Starting feature engineering.")
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    split_date = pd.Timestamp("2024-07-01 00:00:00+00:00")

    try:
        logger.info("Step 1: Loading and standardizing data.")
        df = load_and_standardize_data()

        logger.info("Step 2: Adding time context.")
        df = add_time_context(df)

        logger.info("Step 3: Computing target and errors.")
        df = add_target_and_errors(df)

        logger.info("Step 4: Extracting system-level data.")
        df_system = extract_system_level(df)

        logger.info("Step 5: Engineering features.")
        df_system = engineer_features(df_system)

        logger.info("Step 6: Creating train-test split.")
        df_system = create_train_test_split(df_system, split_date)

        logger.info("Step 7: Building model-ready dataset.")
        model_ready = build_model_ready_dataset(df_system)

        model_ready.to_csv(output_path, index=False)
        logger.info(f"Saved model-ready dataset: {output_path}.")

        logger.info("Feature engineering complete.")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}.")
        raise


if __name__ == "__main__":
    app()
