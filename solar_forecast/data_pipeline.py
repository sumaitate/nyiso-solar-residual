from pathlib import Path
import pandas as pd
from loguru import logger
import typer

from solar_forecast.config import (
    PROCESSED_ROOT,
    SOLAR_RAW_ROOT,
    SOLAR_ZIP_PATH,
    UNZIPPED_ROOTS,
    NYISO_OUT,
    ERA5_OUT,
    MERGED_OUT,
    TS_COL as ts_col,
    ZONE_COL as zone_col,
)

from solar_forecast.dataset import (
    unzip_main_archive,
    unzip_all_archives,
    load_folder,
)

app = typer.Typer()


def extract_and_prepare_nyiso() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    logger.info("Extracting NYISO data.")

    unzip_main_archive(SOLAR_ZIP_PATH, SOLAR_RAW_ROOT)
    unzip_all_archives(SOLAR_RAW_ROOT / "actuals", UNZIPPED_ROOTS["actuals"])
    unzip_all_archives(SOLAR_RAW_ROOT / "forecasts", UNZIPPED_ROOTS["forecasts"])
    unzip_all_archives(SOLAR_RAW_ROOT / "capacity", UNZIPPED_ROOTS["capacity"])

    logger.info("Loading CSVs into DataFrames.")
    df_actual = load_folder(UNZIPPED_ROOTS["actuals"])
    df_forecast = load_folder(UNZIPPED_ROOTS["forecasts"])
    df_capacity = load_folder(UNZIPPED_ROOTS["capacity"])

    logger.info(f"Loaded actuals: {df_actual.shape}.")
    logger.info(f"Loaded forecasts: {df_forecast.shape}.")
    logger.info(f"Loaded capacity: {df_capacity.shape}.")

    for df in (df_actual, df_forecast, df_capacity):
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )

    for df in (df_actual, df_forecast, df_capacity):
        df[ts_col] = pd.to_datetime(df[ts_col], format="%m/%d/%Y %H:%M", errors="coerce")

    for df in (df_actual, df_forecast, df_capacity):
        for col in df.select_dtypes(include="object").columns:
            if col not in ["zone_name", "time_zone"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

    df_actual = df_actual.rename(columns={"mw_value": "actual_mw"})
    df_forecast = df_forecast.rename(columns={"mw_value": "forecast_mw"})
    df_capacity = df_capacity.rename(columns={"mw_value": "capacity_mw"})

    return df_actual, df_forecast, df_capacity


def merge_nyiso_data(
    df_actual: pd.DataFrame,
    df_forecast: pd.DataFrame,
    df_capacity: pd.DataFrame
) -> pd.DataFrame:
    logger.info("Merging NYISO data to system level.")

    df_actual_hourly = (
        df_actual
        .dropna(subset=[ts_col, zone_col, "actual_mw"])
        .groupby([ts_col, zone_col], as_index=False)["actual_mw"]
        .sum()
    )

    df_forecast_hourly = (
        df_forecast
        .dropna(subset=[ts_col, zone_col, "forecast_mw"])
        .groupby([ts_col, zone_col], as_index=False)["forecast_mw"]
        .sum()
    )

    df_capacity_updates = (
        df_capacity
        .dropna(subset=[ts_col, zone_col, "capacity_mw"])
        .sort_values([zone_col, ts_col, "source_file"])
        .groupby([ts_col, zone_col], as_index=False)["capacity_mw"]
        .last()
    )

    logger.info(f"Actual hourly: {df_actual_hourly.shape}.")
    logger.info(f"Forecast hourly: {df_forecast_hourly.shape}.")
    logger.info(f"Capacity updates: {df_capacity_updates.shape}.")

    df_nyiso = (
        df_actual_hourly
        .merge(df_forecast_hourly, how="outer", on=[ts_col, zone_col])
        .sort_values([zone_col, ts_col])
        .reset_index(drop=True)
    )

    df_nyiso = (
        df_nyiso
        .merge(df_capacity_updates, how="left", on=[ts_col, zone_col])
        .sort_values([zone_col, ts_col])
        .reset_index(drop=True)
    )

    df_nyiso["capacity_mw"] = (
        df_nyiso
        .groupby(zone_col)["capacity_mw"]
        .ffill()
    )

    logger.info(f"Merged NYISO shape: {df_nyiso.shape}.")
    logger.info(f"Date range: {df_nyiso[ts_col].min()} to {df_nyiso[ts_col].max()}.")

    return df_nyiso


def prepare_era5_data() -> pd.DataFrame:
    logger.info("Loading ERA5 weather data.")

    if not ERA5_OUT.exists():
        raise FileNotFoundError(
            f"ERA5 data not found at {ERA5_OUT}. "
            "Check data/raw/README.md for instructions."
        )

    df_era5 = pd.read_csv(ERA5_OUT, low_memory=False)

    df_era5.columns = (
        df_era5.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace("-", "_", regex=False)
    )

    if "time_stamp" in df_era5.columns:
        df_era5[ts_col] = pd.to_datetime(df_era5["time_stamp"], utc=True, errors="coerce")
    elif "time" in df_era5.columns:
        df_era5[ts_col] = pd.to_datetime(df_era5["time"], utc=True, errors="coerce")
    else:
        raise KeyError(
            f"Timestamp column not found. Columns: {df_era5.columns.tolist()}."
        )

    df_era5[ts_col] = df_era5[ts_col].dt.floor("h")
    df_era5 = df_era5.dropna(subset=[ts_col])
    df_era5 = df_era5.groupby(ts_col, as_index=False).first()

    logger.info(f"Prepared ERA5 shape: {df_era5.shape}.")

    return df_era5


def merge_all_data(df_nyiso: pd.DataFrame, df_era5: pd.DataFrame) -> pd.DataFrame:
    logger.info("Merging NYISO with ERA5.")

    df_nyiso[ts_col] = pd.to_datetime(df_nyiso[ts_col], utc=True, errors="coerce")

    df_merge = pd.merge(df_nyiso, df_era5, on=ts_col, how="inner")
    df_merge = df_merge.sort_values([ts_col, zone_col]).reset_index(drop=True)

    logger.info(f"Merged shape: {df_merge.shape}.")
    logger.info(f"Date range: {df_merge[ts_col].min()} to {df_merge[ts_col].max()}.")

    return df_merge


@app.command()
def main(
    output_nyiso: Path = NYISO_OUT,
    output_merged: Path = MERGED_OUT,
):
    logger.info("Starting data pipeline.")
    PROCESSED_ROOT.mkdir(parents=True, exist_ok=True)

    try:
        logger.info("Step 1: Extracting and preparing NYISO data.")
        df_actual, df_forecast, df_capacity = extract_and_prepare_nyiso()

        logger.info("Step 2: Merging NYISO data.")
        df_nyiso = merge_nyiso_data(df_actual, df_forecast, df_capacity)
        df_nyiso.to_csv(output_nyiso, index=False)
        logger.info(f"Saved: {output_nyiso}.")

        logger.info("Step 3: Preparing ERA5 weather data.")
        df_era5 = prepare_era5_data()
        df_era5.to_csv(ERA5_OUT, index=False)
        logger.info(f"Saved: {ERA5_OUT}.")

        logger.info("Step 4: Merging all data.")
        df_merge = merge_all_data(df_nyiso, df_era5)
        df_merge.to_csv(output_merged, index=False)
        logger.info(f"Saved: {output_merged}.")

        logger.info("Data pipeline complete.")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}.")
        raise


if __name__ == "__main__":
    app()
