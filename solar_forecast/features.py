"""
features.py
physics-informed feature construction for the nyiso solar residual
learning pipeline. importable by notebooks and by make targets.
all functions operate on df_system — the system-level subset of the
merged nyiso + era5 dataset.
"""

import numpy as np
import pandas as pd

from solar_forecast.config import TS_COL, ZONE_COL, TARGET


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    encode hour, month, and day-of-year as sine/cosine pairs so that
    the model sees hour 23 and hour 0 as adjacent rather than 23 units apart.
    sin(2*pi*t/T) and cos(2*pi*t/T) together uniquely locate any t on a circle.
    """
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
    """
    binary flags for physically distinct solar generation regimes.
    morning ramp (hours 6-9): steep irradiance rise, high cloud sensitivity.
    midday (hours 10-14): peak zenith angle, highest panel efficiency.
    """
    df = df.copy()
    df["is_morning_ramp"] = df["hour_local"].between(6, 9).astype(int)
    df["is_midday"]       = df["hour_local"].between(10, 14).astype(int)
    return df


def add_interact_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    cross-product interaction terms encoding:
      forecast_x_hour_sin/cos — how much nyiso predicted at each diurnal phase
      shortwave_x_cloud       — irradiance attenuation: G * (C/100), beer-lambert proxy
      shortwave_x_temp        — pv temperature correction: efficiency drops ~0.4%/degC above 25C
    """
    df = df.copy()
    df["forecast_x_hour_sin"] = df["forecast_mw"] * df["hour_sin"]
    df["forecast_x_hour_cos"] = df["forecast_mw"] * df["hour_cos"]
    df["shortwave_x_cloud"]   = df["shortwave_radiation"] * (df["cloud_cover"] / 100.0)
    df["shortwave_x_temp"]    = df["shortwave_radiation"] * df["temperature_2m"]
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    lagged rolling means and first differences. all rolling windows are
    computed on .shift(1) to ensure no information from hour t leaks into
    the feature — the window always ends at t-1.
    """
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
