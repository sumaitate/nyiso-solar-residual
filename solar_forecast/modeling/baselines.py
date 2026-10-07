"""Simple residual-correction baseline models."""

import pandas as pd


def fit_hourly_climatology(
    fit_df,
    target_col="forecast_error_mw",
):
    """Fit mean residual by local hour."""
    hourly_map = fit_df.groupby("hour_local")[target_col].mean()
    global_mean = fit_df[target_col].mean()
    return hourly_map, global_mean


def predict_hourly_climatology(
    eval_df,
    hourly_map,
    global_mean,
):
    """Predict residual correction from local hour."""
    residual = (
        eval_df["hour_local"]
        .map(hourly_map)
        .fillna(global_mean)
    )

    return (
        eval_df["forecast_mw"]
        + pd.Series(residual, index=eval_df.index)
    )


def fit_month_hour_climatology(
    fit_df,
    target_col="forecast_error_mw",
):
    """Fit mean residual by local month and hour."""
    month_hour_map = fit_df.groupby(
        ["month_local", "hour_local"]
    )[target_col].mean()

    hourly_map = fit_df.groupby(
        "hour_local"
    )[target_col].mean()

    global_mean = fit_df[target_col].mean()

    return month_hour_map, hourly_map, global_mean


def predict_month_hour_climatology(
    eval_df,
    month_hour_map,
    hourly_map,
    global_mean,
):
    """Predict residual correction with hourly fallbacks."""
    residual = []

    for month, hour in zip(
        eval_df["month_local"],
        eval_df["hour_local"],
    ):
        if (month, hour) in month_hour_map.index:
            value = month_hour_map.loc[(month, hour)]
        elif hour in hourly_map.index:
            value = hourly_map.loc[hour]
        else:
            value = global_mean

        residual.append(value)

    return (
        eval_df["forecast_mw"]
        + pd.Series(residual, index=eval_df.index)
    )
