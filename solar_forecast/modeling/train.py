from pathlib import Path
import pickle
import pandas as pd
from loguru import logger
import typer

from solar_forecast.config import MODEL_READY_OUT, MODEL_ROOT

app = typer.Typer()


def fit_mh_clim(fit_df, target_col="forecast_error_mw"):
    """Fit Month-Hour Residual Climatology model"""
    mh_map = fit_df.groupby(["month_local", "hour_local"])[target_col].mean()
    hour_map = fit_df.groupby("hour_local")[target_col].mean()
    global_mean = fit_df[target_col].mean()
    return mh_map, hour_map, global_mean


@app.command()
def main(
    data_path: Path = MODEL_READY_OUT,
    model_path: Path = MODEL_ROOT / "month_hour_climatology_production.pkl",
):
    """Train Month-Hour Residual Climatology model on full training dataset."""
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
    
    logger.info(f"Data shape: {df.shape}")
    
    train_df = df[df["dataset_split"] == "train"].copy()
    logger.info(f"Training set shape: {train_df.shape}")
    
    logger.info("Fitting Month-Hour Residual Climatology model...")
    mh_map, hour_map, global_mean = fit_mh_clim(train_df, target_col="forecast_error_mw")
    
    model_data = {
        "mh_map": mh_map,
        "hour_map": hour_map,
        "global_mean": global_mean,
        "model_type": "month_hour_climatology",
    }
    
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model_data, f)
    
    logger.success(f"Model saved to {model_path}")
    logger.info(f"Month-Hour combinations: {len(mh_map)}")
    logger.info(f"Hourly combinations: {len(hour_map)}")
    logger.info(f"Global mean residual: {global_mean:.2f} MW")

if __name__ == "__main__":
    app()
