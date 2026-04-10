# Reproducibility Guide

Get started with this project.

## Quick Start

```bash
source .venv/bin/activate
python -m solar_forecast.data_pipeline
python -m solar_forecast.features
python -m solar_forecast.modeling.train
```

## Full Workflow

### Step 1: Set up the environment

```bash
# Clone repo
git clone https://github.com/sumaitate/nyiso-solar-residual.git
cd nyiso-solar-residual

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # or: .\.venv\Scripts\activate (Windows)

# Install dependencies
pip install -e .
```

### Step 2: Data Pipeline

Make sure you have the following files downloaded:
- `data/raw/nyiso_solar.zip`
- `data/processed/02_era5_weather.csv`

Then run:

```bash
python -m solar_forecast.data_pipeline
```

Expected output:
- `data/processed/01_nyiso_merged.csv`
- `data/processed/02_era5_weather.csv`
- `data/processed/03_merged_data.csv`

### Step 3: Exploratory Data Analysis

```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### Step 4: Feature Engineering

```bash
python -m solar_forecast.features
```

### Step 5: Model Training

This trains on the Month-Hour Climatology model as it was the best performing model in this project.

```bash
python -m solar_forecast.modeling.train
```

Expected output:
- `models/month_hour_climatology.pkl`
- `models/model_metrics.csv`
- `models/test_predictions.csv`

### Step 6: Error Analysis

```bash
jupyter notebook notebooks/06_error_analysis.ipynb
```

### Step 7: Generate Predictions

```python
from solar_forecast.inference import SolarForecastPredictor
import pandas as pd

# Load predictor
predictor = SolarForecastPredictor('models/month_hour_climatology.pkl')

# Load features
df = pd.read_csv('data/processed/04_system_model_ready_data.csv')

# Generate predictions
predictions = predictor.predict(df)

# Correct NYISO forecasts
df_nyiso = pd.read_csv('data/processed/01_nyiso_merged.csv')
corrected = predictor.correct_forecast(df_nyiso, predictions)
```

## Expected Results

Best model: Month-Hour Residual Climatology

| Metric | Value |
|--------|-------|
| MAE improvement | 5.4% |
| Daytime MAE | 101.53 MW |
| Peak improvement (noon-3 PM) | 24 MW |

## Data Splits

- **Training:** Nov 2020 - Jun 30 2024 (26,627 records)
- **Validation:** Jan 1 - Jun 30 2024 (4,295 records)
- **Test:** Jul 1 2024 - Sep 2025 (10,529 records)

## Troubleshooting

**"ModuleNotFoundError: No module named 'solar_forecast'"**

```bash
pip install -e .
```

**"FileNotFoundError: nyiso_solar.zip"**

- Check `data/raw/` has the file
- See `data/raw/README.md`

**"Data pipeline is slow"**

- Normal for 5M+ rows
- Takes 2-3 minutes total

## For Developers

See docstrings in `solar_forecast/` modules for API details:

- `data_pipeline.py` - Data orchestration
- `inference.py` - Predictions and corrections
- `features.py` - Feature engineering
- `dataset.py` - Data loading utilities
