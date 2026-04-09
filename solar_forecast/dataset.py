"""
Extract, Load, and Parse Raw NYISO Solar and Weather Data
"""

import os
import zipfile
from pathlib import Path
import pandas as pd
from loguru import logger

from solar_forecast.config import TS_COL, ZONE_COL

def unzip_main_archive(zip_path: Path, output_root: Path) -> None:
    if zip_path.exists():
        output_root.mkdir(parents=True, exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(output_root)
            logger.info(f"extracted main: {zip_path}")
        except Exception as e:
            logger.error(f"didn't extract main: {zip_path.name} | {e}")
    else:
        logger.warning(f"not found: {zip_path}")


def unzip_all_archives(input_folder: Path, output_folder: Path) -> None:
    os.makedirs(output_folder, exist_ok=True)
    extracted = 0

    if not input_folder.exists():
        logger.warning(f"input folder not found: {input_folder}")
        return

    for filename in os.listdir(input_folder):
        if filename.endswith(".zip"):
            try:
                with zipfile.ZipFile(input_folder / filename, "r") as archive:
                    archive.extractall(output_folder)
                    extracted += 1
            except Exception as e:
                logger.error(f"did not extract: {filename} | {e}")

    logger.info(f"extraction completed: {extracted} archives from {input_folder}")


def load_folder(folder: Path) -> pd.DataFrame:
    csv_files = list(folder.glob("*.csv"))
    frames = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)
            df["source_file"] = file.name
            frames.append(df)
        except Exception as e:
            logger.error(f"failed to read: {file.name} | {e}")

    if not frames:
        logger.warning(f"no csv files found in: {folder}")
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)


def ensure_required_columns(df: pd.DataFrame, df_name: str) -> None:
    missing = [col for col in [TS_COL, ZONE_COL] if col not in df.columns]
    if missing:
        raise KeyError(
            f"{df_name} is missing required columns: {missing}. "
            f"found: {df.columns.tolist()}"
        )


def resolve_value_col(df: pd.DataFrame) -> str:
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
        f"no megawatts column found. available columns: {df.columns.tolist()}"
    )


def parse_nyiso_time(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    raw_ts    = pd.to_datetime(df[TS_COL], errors="coerce")
    parsed_ts = pd.Series(pd.NaT, index=df.index, dtype="object")

    if "time_zone" in df.columns:
        tz_series  = df["time_zone"].astype(str).str.upper().str.strip()

        is_est     = tz_series.eq("EST")
        is_edt     = tz_series.eq("EDT")
        other_mask = ~(is_est | is_edt)

        if is_est.any():
            parsed_ts.loc[is_est] = (
                pd.to_datetime(df.loc[is_est, TS_COL], errors="coerce")
                .dt.tz_localize("Etc/GMT+5", nonexistent="shift_forward", ambiguous="NaT")
                .dt.tz_convert("UTC")
            )

        if is_edt.any():
            parsed_ts.loc[is_edt] = (
                pd.to_datetime(df.loc[is_edt, TS_COL], errors="coerce")
                .dt.tz_localize("Etc/GMT+4", nonexistent="shift_forward", ambiguous="NaT")
                .dt.tz_convert("UTC")
            )

        if other_mask.any():
            parsed_ts.loc[other_mask] = (
                pd.to_datetime(df.loc[other_mask, TS_COL], errors="coerce")
                .dt.tz_localize(
                    "America/New_York",
                    nonexistent="shift_forward",
                    ambiguous="NaT",
                )
                .dt.tz_convert("UTC")
            )
    else:
        parsed_ts = (
            raw_ts
            .dt.tz_localize(
                "America/New_York",
                nonexistent="shift_forward",
                ambiguous="NaT",
            )
            .dt.tz_convert("UTC")
            .astype("object")
        )

    df[TS_COL] = pd.to_datetime(parsed_ts, utc=True, errors="coerce").dt.floor("h")

    df[ZONE_COL] = (
        df[ZONE_COL]
        .astype(str)
        .str.upper()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    return df
