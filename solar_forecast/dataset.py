"""
dataset.py
Functions for extracting raw NYISO archives and loading CSV folders
into DataFrames. Importable by notebooks and by make data targets.
"""

import os
import zipfile
from pathlib import Path

import pandas as pd
from loguru import logger

from solar_forecast.config import TS_COL, ZONE_COL


def unzip_main_archive(zip_path: Path, output_root: Path) -> None:
    """Extract the top-level NYISO solar ZIP into output_root."""
    if zip_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(output_root)
            logger.info(f"Extracted Main: {zip_path}")
        except Exception as e:
            logger.error(f"Didn't Extract Main: {zip_path.name} | {e}")
    else:
        logger.warning(f"Not Found: {zip_path}")


def unzip_all_archives(input_folder: Path, output_folder: Path) -> None:
    """Extract every ZIP file in input_folder into output_folder."""
    os.makedirs(output_folder, exist_ok=True)
    extracted = 0

    if not input_folder.exists():
        logger.warning(f"Input folder not found: {input_folder}")
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(input_folder / filename, "r") as archive:
                    archive.extractall(output_folder)
                    extracted += 1
            except Exception as e:
                logger.error(f"Did Not Extract: {filename} | {e}")

    logger.info(f"Extraction Completed: {extracted} archives from {input_folder}")


def load_folder(folder: Path) -> pd.DataFrame:
    """
    Read every CSV in folder into a single concatenated DataFrame.
    Appends source_file column for traceability.
    Returns an empty DataFrame if no CSVs are found.
    """
    csv_files = list(folder.glob("*.csv"))
    frames = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed to Read: {file.name} | {e}")

    if not frames:
        logger.warning(f"No CSV files found in: {folder}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ensure_required_columns(df: pd.DataFrame, df_name: str) -> None:
    """Raise KeyError if time_stamp or zone_name are missing."""
    missing = [col for col in [TS_COL, ZONE_COL] if col not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}. "
            f"Found: {df.columns.tolist()}"
        )


def resolve_value_col(df: pd.DataFrame) -> str:
    """
    Return the name of the megawatt value column by checking a
    priority list of known candidates, then falling back to the
    first numeric column that is not a key or metadata column.
    """
    candidates = [
        "mw_value", "mw", "value",
        "actual_mw", "forecast_mw", "capacity_mw", "name",
    ]
    for col in candidates:
        if col in df.columns:
            return col

    numeric_candidates = []
    for col in df.columns:
        if col in [TS_COL, ZONE_COL, "source_file"]:
            continue
        if pd.to_numeric(df[col], errors="coerce").notna().sum() > 0:
            numeric_candidates.append(col)

    if numeric_candidates:
        return numeric_candidates[0]

    raise KeyError(
        f"No megawatts column found. Available columns: {df.columns.tolist()}"
    )
