from pathlib import Path
import pickle
import pandas as pd
from loguru import logger
import typer

from solar_forecast.config import MODEL_READY_OUT, MODEL_ROOT

app = typer.Typer()


def predict_mh_clim(eval_df, mh_map, hour_map, global_mean):
    """Generate predictions using Month-Hour Residual Climatology."""
    residual = []
    for m, h in zip(eval_df["month_local"], eval_df["hour_local"]):
        if (m, h) in mh_map.index:
            residual.append(mh_map.loc[(m, h)])
        elif h in hour_map.index:
            residual.append(hour_map.loc[h])
        else:
            residual.append(global_mean)
    corrected = eval_df["forecast_mw"] + pd.Series(residual, index=eval_df.index)
    return corrected.clip(lower=0.0)


@app.command()
def main(
    data_path: Path = MODEL_READY_OUT,
    model_path: Path = MODEL_ROOT / "month_hour_climatology_production.pkl",
    predictions_path: Path = MODEL_ROOT / "predictions.csv",
    split: str = "test",
):
    """Generate predictions using trained Month-Hour Residual Climatology model."""
    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path, low_memory=False)
    
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )
    
    df["time_stamp"] = pd.to_datetime(df["time_stamp"], utc=True, errors="coerce")
    df["time_local"] = df["time_stamp"].dt.tz_convert("America/New_York")
    df["hour_local"] = df["time_local"].dt.hour
    df["month_local"] = df["time_local"].dt.month
    
    eval_df = df[df["dataset_split"] == split].copy()
    logger.info(f"Evaluation set ({split}) shape: {eval_df.shape}")
    
    logger.info(f"Loading model from {model_path}...")
    with open(model_path, "rb") as f:
        model_data = pickle.load(f)
    
    logger.info("Generating predictions...")
    corrected_forecast = predict_mh_clim(
        eval_df,
        model_data["mh_map"],
        model_data["hour_map"],
        model_data["global_mean"],
    )
    
    results_df = eval_df[[
        "time_stamp",
        "time_local",
        "actual_mw",
        "forecast_mw",
        "hour_local",
        "month_local",
    ]].copy()
    
    results_df["corrected_forecast_mw"] = corrected_forecast
    results_df["baseline_error_mw"] = results_df["actual_mw"] - results_df["forecast_mw"]
    results_df["model_error_mw"] = results_df["actual_mw"] - results_df["corrected_forecast_mw"]
    results_df["baseline_abs_error"] = results_df["baseline_error_mw"].abs()
    results_df["model_abs_error"] = results_df["model_error_mw"].abs()
    
    results_df.to_csv(predictions_path, index=False)
    logger.success(f"Predictions saved to {predictions_path}")
    
    mae_baseline = results_df["baseline_abs_error"].mean()
    mae_model = results_df["model_abs_error"].mean()
    improvement = mae_baseline - mae_model
    
    logger.info(f"Baseline MAE: {mae_baseline:.2f} MW")
    logger.info(f"Model MAE: {mae_model:.2f} MW")
    logger.info(f"Improvement: {improvement:.2f} MW ({100*improvement/mae_baseline:.2f}%)")

if __name__ == "__main__":
    app()
