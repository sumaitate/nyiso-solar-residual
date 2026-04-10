import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger


class SolarForecastPredictor:
    def __init__(self, model_path: Path):
        logger.info(f"Loading model from {model_path}.")
        
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found at {model_path}.")
        
        with open(model_path, "rb") as f:
            self.model_data = pickle.load(f)
        
        self.mh_map = self.model_data.get("mh_map")
        self.hour_map = self.model_data.get("hour_map")
        self.global_mean = self.model_data.get("global_mean")
        
        logger.info("Model loaded successfully.")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        df = df.copy()
        
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
        
        predictions = []
        
        for idx, row in df.iterrows():
            month = row["month_local"]
            hour = row["hour_local"]
            
            if (month, hour) in self.mh_map.index:
                pred = self.mh_map.loc[(month, hour)]
            elif hour in self.hour_map.index:
                pred = self.hour_map.loc[hour]
            else:
                pred = self.global_mean
            
            predictions.append(pred)
        
        logger.info(f"Generated {len(predictions)} predictions.")
        return pd.Series(predictions, index=df.index)

    def correct_forecast(
        self,
        df_nyiso: pd.DataFrame,
        predictions: pd.Series
    ) -> pd.DataFrame:
        df_nyiso = df_nyiso.copy()
        
        df_nyiso.columns = (
            df_nyiso.columns.str.strip()
            .str.lower()
            .str.replace(" ", "_", regex=False)
            .str.replace("-", "_", regex=False)
        )
        
        df_nyiso["time_stamp"] = pd.to_datetime(
            df_nyiso["time_stamp"],
            utc=True,
            errors="coerce"
        )
        
        if len(predictions) != len(df_nyiso):
            logger.warning(
                f"Prediction length {len(predictions)} != "
                f"NYISO forecast length {len(df_nyiso)}. "
                "Truncating to shorter length."
            )
            min_len = min(len(predictions), len(df_nyiso))
            df_nyiso = df_nyiso.iloc[:min_len].reset_index(drop=True)
            predictions = predictions.iloc[:min_len].reset_index(drop=True)
        
        df_nyiso["residual_correction_mw"] = predictions.values
        df_nyiso["corrected_forecast_mw"] = (
            df_nyiso["forecast_mw"] + df_nyiso["residual_correction_mw"]
        )
        
        df_nyiso["corrected_forecast_mw"] = df_nyiso["corrected_forecast_mw"].clip(lower=0)
        
        logger.info(
            f"Applied corrections to {len(df_nyiso)} forecasts. "
            f"Mean correction: {predictions.mean():.2f} MW."
        )
        
        return df_nyiso


if __name__ == "__main__":
    pass
