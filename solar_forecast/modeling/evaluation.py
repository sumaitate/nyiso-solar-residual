"""Shared model evaluation utilities."""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def rmse(y_true, y_pred):
    """Return root mean squared error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def score_model(name, actual, forecast, day_mask):
    """Compute overall and daylight evaluation metrics."""
    return {
        "Model": name,
        "MAE": mean_absolute_error(actual, forecast),
        "RMSE": rmse(actual, forecast),
        "Daylight_MAE": mean_absolute_error(
            actual.loc[day_mask],
            forecast.loc[day_mask],
        ),
        "Daylight_RMSE": rmse(
            actual.loc[day_mask],
            forecast.loc[day_mask],
        ),
    }


def selection_score(actual, forecast, day_mask):
    """Compute metrics used for validation-based model selection."""
    mae = mean_absolute_error(actual, forecast)
    rmse_value = rmse(actual, forecast)

    day_mae = mean_absolute_error(
        actual.loc[day_mask],
        forecast.loc[day_mask],
    )
    day_rmse = rmse(
        actual.loc[day_mask],
        forecast.loc[day_mask],
    )

    return {
        "MAE": mae,
        "RMSE": rmse_value,
        "Daylight_MAE": day_mae,
        "Daylight_RMSE": day_rmse,
        "Selection_Score": (
            0.40 * day_mae
            + 0.30 * mae
            + 0.20 * day_rmse
            + 0.10 * rmse_value
        ),
    }


def clip_zero(series):
    """Clip forecasts at zero MW."""
    return pd.Series(series, index=series.index).clip(lower=0.0)


def build_prediction_frame(
    model_name,
    eval_df,
    corrected_forecast,
    ts_col="time_stamp",
):
    """Build the standard prediction/error table."""
    pred = eval_df[
        [
            ts_col,
            "time_local",
            "actual_mw",
            "forecast_mw",
            "hour_local",
            "month_local",
            "is_daylight",
        ]
    ].copy()

    pred["model_name"] = model_name
    pred["corrected_forecast_mw"] = clip_zero(corrected_forecast)

    pred["baseline_error_mw"] = (
        pred["actual_mw"] - pred["forecast_mw"]
    )
    pred["model_error_mw"] = (
        pred["actual_mw"] - pred["corrected_forecast_mw"]
    )

    pred["baseline_abs_error"] = pred["baseline_error_mw"].abs()
    pred["model_abs_error"] = pred["model_error_mw"].abs()

    return pred


def add_improvement_cols(
    results_df,
    baseline_name="NYISO Baseline",
):
    """Add metric improvements relative to the baseline."""
    results_df = results_df.copy()

    base = results_df.loc[
        results_df["Model"] == baseline_name
    ].iloc[0]

    results_df["MAE_Improvement_vs_NYISO"] = (
        base["MAE"] - results_df["MAE"]
    )
    results_df["RMSE_Improvement_vs_NYISO"] = (
        base["RMSE"] - results_df["RMSE"]
    )
    results_df["Daylight_MAE_Improvement_vs_NYISO"] = (
        base["Daylight_MAE"] - results_df["Daylight_MAE"]
    )
    results_df["Daylight_RMSE_Improvement_vs_NYISO"] = (
        base["Daylight_RMSE"] - results_df["Daylight_RMSE"]
    )

    return results_df


def summarize_errors(df):
    """Summarize baseline and corrected forecast errors."""
    return pd.Series(
        {
            "n_obs": len(df),
            "baseline_mae": df["baseline_abs_error"].mean(),
            "model_mae": df["model_abs_error"].mean(),
            "baseline_rmse": np.sqrt(
                df["baseline_sq_error"].mean()
            ),
            "model_rmse": np.sqrt(
                df["model_sq_error"].mean()
            ),
            "baseline_bias": df["baseline_error_mw"].mean(),
            "model_bias": df["model_error_mw"].mean(),
            "improved_share": df["improved_flag"].mean(),
            "worsened_share": df["worsened_flag"].mean(),
        }
    )


def add_reduction_cols(df):
    """Add absolute and percentage reductions in error."""
    df = df.copy()

    df["mae_reduction"] = (
        df["baseline_mae"] - df["model_mae"]
    )
    df["rmse_reduction"] = (
        df["baseline_rmse"] - df["model_rmse"]
    )

    df["mae_reduction_pct"] = (
        100 * df["mae_reduction"] / df["baseline_mae"]
    )
    df["rmse_reduction_pct"] = (
        100 * df["rmse_reduction"] / df["baseline_rmse"]
    )

    return df
