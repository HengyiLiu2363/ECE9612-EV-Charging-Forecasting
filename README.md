# ECE9612-EV-Charging-Forecasting
ECE9612 final project on EV charging station usage analysis, demand forecasting, and usage pattern modeling using machine learning.
# ECE9612 EV Charging Forecasting Project

## Objective
This project studies EV charging station usage data and develops machine learning models for charging demand forecasting and usage pattern analysis.

## Team Members
- Salar Bugti
- Junze Shan
- Shengning Wang
- Ruoyin Wang
- Hengyi Liu

## Project Scope
The main objective is to forecast EV charging demand from historical charging session data.  
The project also explores charging usage patterns through classification and clustering analysis.

## Repository Structure
- `data/` : raw, interim, and processed data files stored locally
- `notebooks/` : exploratory analysis and development notebooks
- `src/` : reusable Python modules for data processing, features, models, and visualization
- `scripts/` : runnable scripts for preprocessing and forecasting
- `reports/` : figures and tables for the final report
- `slides/` : presentation materials

## Main Technical Workflow
1. Load and inspect EV charging session data
2. Clean and preprocess the raw dataset
3. Aggregate charging sessions into hourly demand targets
4. Build time-based and lag-based features
5. Train and compare forecasting models
6. Evaluate model performance using regression metrics

## Initial Forecasting Models
- Persistence baseline
- Historical average baseline
- Linear Regression
- Random Forest Regressor
- Gradient Boosting / XGBoost

## Setup
1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Place raw dataset files in `data/raw/`
5. Run preprocessing and forecasting scripts

## Notes
Large raw data files and generated processed outputs are not committed to GitHub.