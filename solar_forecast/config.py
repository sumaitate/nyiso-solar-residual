"""Project Configuration and Path Management"""

from pathlib import Path
from loguru import logger

PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_ROOT      = PROJ_ROOT / "data"
RAW_ROOT       = DATA_ROOT / "raw"
INTERIM_ROOT   = DATA_ROOT / "interim" 
PROCESSED_ROOT = DATA_ROOT / "processed"
EXTERNAL_ROOT  = DATA_ROOT / "external"

SOLAR_RAW_ROOT = RAW_ROOT / "nyiso_solar"
SOLAR_ZIP_PATH = RAW_ROOT / "nyiso_solar.zip"

UNZIPPED_ROOTS = {
    "actuals":   SOLAR_RAW_ROOT / "unzipped_actuals",
    "forecasts": SOLAR_RAW_ROOT / "unzipped_forecasts",
    "capacity":  SOLAR_RAW_ROOT / "unzipped_capacity",

}

NYISO_OUT  = PROCESSED_ROOT / "01_nyiso_merged.csv"
ERA5_OUT   = PROCESSED_ROOT / "02_era5_weather.csv"
MERGED_OUT = PROCESSED_ROOT / "03_merged_data.csv"

MODEL_READY_OUT = PROCESSED_ROOT / "04_system_model_ready_data.csv"

MODEL_ROOT = PROJ_ROOT / "models"
REPORTS_ROOT  = PROJ_ROOT / "reports"
FIGURES_ROOT  = REPORTS_ROOT / "figures"

TS_COL   = "time_stamp"
ZONE_COL = "zone_name"
TARGET   = "forecast_error_mw"
