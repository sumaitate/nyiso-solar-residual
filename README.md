# Physics-Informed Residual Learning for Solar Forecasting

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

The New York Independent System Operator (NYISO) releases solar forecasts, but these contain errors. This project uses machine learning with weather features to predict and correct those forecast errors for a more accurate forecast.

---
## Overview

### Problem Statement
NYISO produces baseline solar forecasts for the electric grid, but these forecasts contain systematic errors that can reduce grid reliability and increase operational costs. When forecasts deviate significantly from actual solar generation, grid operators have to deploy expensive backup resources or risk blackouts. This project addresses whether machine learning can minimize forecast errors by learning patterns in the difference between what NYISO predicts and what actually happens, rather than trying to predict solar generation from scratch.

### Research Question
Can a machine learning model trained on forecast residuals combined with ERA5 weather features produce more accurate predictions than the baseline NYISO forecast alone?

### Key Contributions
* Month-Hour Residual Climatology model outperforms ML baselines
* 5.4% MAE improvement over NYISO baseline on held-out test set
* Analysis of when/where errors occur (daytime > nighttime, summer > winter)

---
## Technical Approach

### Data Sources and Scope
Data comes from public NYISO and ERA5 sources (Nov 2020–Sep 2025), containing 509,364 
hourly records across 11 solar zones plus system-level aggregation. Due to geographic 
data constraints, analysis focuses on SYSTEM-level data representing total New York 
solar generation rather than individual zones with varying installed capacities.

The merged dataset includes 12 columns: timestamps, generation in megawatts, and 8 
weather variables (temperature, pressure, cloud cover, wind speed, shortwave radiation). 
Missing data rates are low (1.7% in generation, <0.7% in forecasts).

The original data processing pipeline is in notebooks/ but not currently documented. 
A formal reproducibility guide will be added in a future update.

### Methodology
The main strategy models forecast errors as residuals—the difference between actual solar generation and NYISO's  forecast—rather than predicting generation from scratch. This residual becomes the target variable that  models learn to predict.

#### Feature Engineering
In the feature engineering step, 25 physics-informed features capture solar generation patterns. The use of sine and cosine transformations preserve temporal continuity by hourly phase (24-hour cycle), month (12-month cycle), and day of year (365.25-day cycle). These features interact with weather features such as: shortwave radiation multiplied by cloud cover, shortwave radiation multiplied by temperature, and forecast magnitude multiplied by hourly phase. Rolling averages capture short-term momentum using three-hour and 24-hour windows of both forecast and radiation. Change features measure volatility through first-order differences and absolute ramps in radiation. Binary regime flags identify morning ramp periods (6–9 AM) and midday hours (10–2 PM), when forecast errors are systematically largest.

#### Data Splitting
The dataset splits at July 1, 2024: 30,922 training records before that date and 
10,529 test records after. Within training data, a secondary split at January 1, 2024 
creates 26,627 sub-training records and 4,295 validation records for hyperparameter 
tuning and selection.

#### Model Selection and Training
Six model classes are evaluated: simple climatological baselines (mean residual, 
hourly climatology, and month-hour climatology), Ridge regression, and 4 gradient 
boosting variants (LightGBM, XGBoost, CatBoost, HistGradientBoosting). Tree models 
undergo grid search over learning rate, tree depth, regularization, and subsampling. 
All models are trained on subtraining data, optimized on validation data, and refitted 
on the full training set for final evaluation.

### Validation and Testing

#### Evaluation Metrics
Performance is evaluated using MAE and RMSE, with separate metrics for overall and daylight-only periods. Daylight hours (shortwave radiation above zero) represent 55% of test records and are weighted more heavily because forecast errors during peak generation directly impact grid operations. A selection score combines daytime MAE (40% weight), overall MAE (30%), daytime RMSE (20%), and overall RMSE (10%).

#### Best Performing Model
The best-performing model is Month-Hour Residual Climatology, a simple statistical approach that computes average residuals grouped by month and hour from training data. When a specific month-hour pair lacks data, it falls back to hourly average, then to global mean. This method outperforms all machine learning models on the validation set (103.23 MW MAE vs. 103.87 MW for NYISO baseline) and on the held-out test set.

---
## Results
### Overall Performance
The Month-Hour Residual Climatology model achieves a 5.4% reduction in MAE compared to NYISO's baseline on the test set, reducing MAE from 106.95 MW to 101.14 MW and RMSE from 207.75 MW to 200.79 MW. During daylight hours, improvements are more pronounced: daytime MAE decreases by 6.61 MW (5.3%) and daytime RMSE by 7.46 MW (3.3%).

#### Temporal Patterns
Error reduction is not uniform across time. Peak improvements occur between noon and 3 PM (24 MW MAE, 30 MW RMSE reduction). Morning hours (6–10 AM) show minimal improvement (1–4 MW) because forecast errors during ramping periods are harder to predict. Midday hours (10 AM–2 PM) show strong improvement (11–25 MW), indicating the model learns how NYISO systematically underestimates during peak generation.

#### Weather Dependency
Performance varies significantly by weather conditions. High-irradiance periods (381–965 W/m²) see 14.3 MW MAE reduction versus 3.7 MW in low-irradiance periods. Clear-sky conditions show 6.5 MW reduction compared to 4.8 MW in high-cloud-cover (64–100%). Cold periods (below −0.8°C) see 7.1 MW reduction; warmer periods see 5.8–11.1 MW.

#### Seasonal Patterns
MAE improves in 11 of 12 months. February and September show the strongest gains (13.8 MW each). Summer months (July–September) consistently reduce MAE by 9.3–13.8 MW, while winter months show variable performance (1.2 MW in January to 9.3 MW in February). Overall, the model improves 35% of individual hourly forecasts, worsens 24%, and has no effect on 41%.

### Limitations and Caveats
The model uses only system-level aggregated data rather than zone-by-zone forecasts because individual zones have different installed capacities and geographic distributions. The model is trained and tested entirely on NYISO data from a single U.S. region, limiting generalizability to other grid operators or solar markets with different forecast methodologies or climates.

Calendar features are limited to hour, month, and day of year; the model doesn't account for holidays, solar events, or multi-year trends. 

Weather features come from ERA5 reanalysis at coarse spatial resolution, which may not capture local variations critical for facility-specific forecasts.

The Month-Hour Climatology approach relies on historical averages and can't adapt to novel weather patterns or unprecedented system states outside the training distribution. Forecast errors during morning and evening ramp periods (6–10 AM and 4–6 PM) remain difficult to minimize, suggesting residual learning alone cannot capture rapid cloud cover or atmospheric changes. The model shows no improvement during nighttime hours when solar output is zero, as residual errors are zero by definition during darkness.

### Interpretation

#### Learnable Forecast Bias
The Month-Hour Climatology model's success over other more complex modeling approaches indicates that NYISO's forecast errors follow primarily seasonal and diurnal patterns rather than complex weather-dependent dynamics. The strong midday improvement (11–25 MW MAE reduction) reflects a consistent systematic underestimation during peak generation hours that the climatology approach addresses and corrects.

#### Limitations of Residual Learning
The minimal improvement during morning and evening ramps (1–4 MW) suggests that rapid atmospheric changes in these times exceed the explanatory power of historical averages. The model cannot adapt to unprecedented cloud cover or temperature variations outside the training distribution, indicating that ramp-period errors are driven by transient weather phenomena rather than just systematic biases.

#### Seasonal Structure
The major improvements in summer and fall (9.3–13.8 MW) compared to variable performance in winter (1.2–9.3 MW) align with solar generation seasonality and atmospheric stability. This pattern suggests NYISO's forecast methodology contains region-specific or seasonal biases that are recoverable through historical aggregation when generation is large and patterns are more stable.

#### Operational Significance
The 5.4% MAE reduction (5.8 MW absolute error on a 584 MW system average) is modest in percentage terms but concentrated during high-generation hours when reserve costs and grid stress are greatest. The model improves 35% of predictions while degrading only 24%, suggesting that selective application—using corrected forecasts when error magnitude is largest and reverting to NYISO baseline otherwise—could provide practical operational benefits without systematic performance loss.

---
## Repository Structure

```
├── LICENSE            
├── Makefile           
├── README.md          <- You are here.
├── data
│   ├── external       
│   ├── interim        
│   ├── processed      
│   └── raw           
│
├── docs               
├── models          
├── notebooks  
│   ├── 00_data_acquisition
│   ├── 01_exploratory_data_analysis
│   ├── 02_feature_engineering
│   ├── 03_baseline_models
│   ├── 04_advanced_models
│   ├── 05_hyperparameter_tuning
│   └── 06_error_analysis
│
├── pyproject.toml  
├── references  
│
├── reports   <- View final report and PowerPoint here.
│   └── figures 
│
├── requirements.txt 
├── setup.cfg    
│
└── solar_forecast   <- src for use in this project.
    ├── __init__.py     
    ├── config.py            
    ├── dataset.py              
    ├── features.py           
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py        
    │   └── train.py   
    └── plots.py          
```
