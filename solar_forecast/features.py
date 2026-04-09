"""
Construct Physics-Informed Features 
"""

import numpy as np
import pandas as pd

from solar_forecast.config import TS_COL, ZONE_COL, TARGET


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dayofyear_local"] = df["time_local"].dt.dayofyear

    df["hour_sin"]      = np.sin(2 * np.pi * df["hour_local"]      / 24)
    df["hour_cos"]      = np.cos(2 * np.pi * df["hour_local"]      / 24)
    df["month_sin"]     = np.sin(2 * np.pi * df["month_local"]     / 12)
    df["month_cos"]     = np.cos(2 * np.pi * df["month_local"]     / 12)
    df["dayofyear_sin"] = np.sin(2 * np.pi * df["dayofyear_local"] / 365.25)
    df["dayofyear_cos"] = np.cos(2 * np.pi * df["dayofyear_local"] / 365.25)

    return df


def add_regime_flags(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["is_morning_ramp"] = df["hour_local"].between(6, 9).astype(int)
    df["is_midday"]       = df["hour_local"].between(10, 14).astype(int)
    return df


def add_interact_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_x_hour_sin"] = df["forecast_mw"] * df["hour_sin"]
    df["forecast_x_hour_cos"] = df["forecast_mw"] * df["hour_cos"]
    df["shortwave_x_cloud"]   = df["shortwave_radiation"] * (df["cloud_cover"] / 100.0)
    df["shortwave_x_temp"]    = df["shortwave_radiation"] * df["temperature_2m"]
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["forecast_roll_mean_3"]   = df["forecast_mw"].shift(1).rolling(3,  min_periods=1).mean()
    df["shortwave_roll_mean_3"]  = df["shortwave_radiation"].shift(1).rolling(3,  min_periods=1).mean()
    df["forecast_roll_mean_24"]  = df["forecast_mw"].shift(1).rolling(24, min_periods=1).mean()
    df["shortwave_roll_mean_24"] = df["shortwave_radiation"].shift(1).rolling(24, min_periods=1).mean()
    df["forecast_diff_1"]        = df["forecast_mw"].diff(1)
    df["shortwave_diff_1"]       = df["shortwave_radiation"].diff(1)
    df["shortwave_ramp_abs"]     = df["shortwave_diff_1"].abs()
    return df


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
