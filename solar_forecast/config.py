
from pathlib import Path

from loguru import logger

# --- Project Root ---

# Resolved relative to this file so it works on any machine

# regardless of absolute path, matching the CCDS convention.

PROJ_ROOT = Path(__file__).resolve().parents[1]

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# --- Data Layer ---

DATA_ROOT      = PROJ_ROOT / "data"

RAW_ROOT       = DATA_ROOT / "raw"

INTERIM_ROOT   = DATA_ROOT / "interim"    # available for intermediate outputs

PROCESSED_ROOT = DATA_ROOT / "processed"

EXTERNAL_ROOT  = DATA_ROOT / "external"

# --- NYISO Raw Subdirectory ---

SOLAR_RAW_ROOT = RAW_ROOT / "nyiso_solar"

SOLAR_ZIP_PATH = RAW_ROOT / "nyiso_solar.zip"

UNZIPPED_ROOTS = {

    "actuals":   SOLAR_RAW_ROOT / "unzipped_actuals",

    "forecasts": SOLAR_RAW_ROOT / "unzipped_forecasts",

    "capacity":  SOLAR_RAW_ROOT / "unzipped_capacity",

}

# paths

NYISO_OUT  = PROCESSED_ROOT / "01_nyiso_merged.csv"

ERA5_OUT   = PROCESSED_ROOT / "02_era5_weather.csv"

MERGED_OUT = PROCESSED_ROOT / "03_merged_data.csv"

MODEL_READY_OUT = PROCESSED_ROOT / "04_system_model_ready_data.csv"

# output

MODEL_ROOT = PROJ_ROOT / "models"

# reports

REPORTS_ROOT  = PROJ_ROOT / "reports"

FIGURES_ROOT  = REPORTS_ROOT / "figures"

# column names

TS_COL   = "time_stamp"

ZONE_COL = "zone_name"

TARGET   = "forecast_error_mw"

