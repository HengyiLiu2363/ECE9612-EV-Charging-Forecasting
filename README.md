# ECE9612 EV Charging Forecasting Project

ECE9612 final project on EV charging station usage analysis, demand forecasting, and usage pattern modeling using machine learning.

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

## Dataset
The project uses ACN EV charging session data collected from the Caltech charging network.  
The raw data is fetched through the ACN API, then processed into session-level and hourly forecasting datasets.

## Repository Structure
- `data/` : raw, interim, and processed data files stored locally
- `notebooks/` : exploratory analysis and development notebooks
- `src/` : reusable Python modules for data processing, feature engineering, and modeling
- `scripts/` : runnable project scripts
- `reports/` : figures and tables for the final report
- `slides/` : presentation materials

## Main Technical Workflow
1. Acquire EV charging session data from the ACN API
2. Clean and preprocess the raw dataset
3. Flatten session data into a structured tabular format
4. Aggregate charging sessions into hourly demand targets
5. Split the hourly dataset into train, validation, and test sets
6. Build forecasting models and compare performance
7. Evaluate models using regression metrics and visualizations

## Forecasting Pipeline Status
Completed:
- GitHub repository setup
- ACN API workflow
- Raw dataset acquisition
- Processed session-level dataset
- Continuous hourly forecasting dataset
- Chronological train/validation/test split
- Baseline forecast

In progress:
- Feature engineering for forecasting
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- LSTM

## Model Scope
The main forecasting target is hourly EV charging session count.

Forecasting models to be developed:
- Baseline forecast
- Linear Regression
- Random Forest Regressor
- Gradient Boosting Regressor
- LSTM

Optional extension:
- Hourly total energy delivered (`total_kwh`) forecasting

## Team Role Division

### Hengyi Liu
**Forecasting pipeline lead**
- ACN API data acquisition
- Raw data preprocessing
- Session-level and hourly dataset construction
- Chronological train/validation/test split
- Forecasting models:
  - Baseline forecast
  - Linear Regression
  - Random Forest Regressor
  - Gradient Boosting Regressor
  - LSTM
- Forecasting model evaluation and comparison
- Main forecasting figures for the report and presentation

### Team Member 2
**EDA and visualization lead**
- Exploratory data analysis
- Temporal usage trends
- Weekday vs weekend patterns
- Monthly and hourly charging behavior
- Supporting data visualizations for report and slides

### Team Member 3
**Classification task lead**
- Define charging behavior classes
- Build classification models for charging duration or energy categories
- Evaluate classification performance
- Summarize behavior prediction results

### Team Member 4
**Clustering and usage pattern modeling lead**
- Unsupervised learning for charging behavior grouping
- K-means clustering
- Cluster interpretation and visualization
- Discussion of customer charging profiles

### Team Member 5
**Report and presentation integration lead**
- Merge written sections into final report
- Organize references, tables, and figures
- Ensure slide and report consistency
- Final formatting and presentation structure

## Repository Workflow
- `main` is the stable shared branch
- Each team member should create and use their own branch
- Work should be committed and pushed from personal branches before being merged into `main`

Example branches:
- `henry-forecasting`
- `member2-eda`
- `member3-classification`
- `member4-clustering`
- `member5-report`

## Setup
1. Clone the repository
2. Create and activate a Python virtual environment
3. Install dependencies with `pip install -r requirements.txt`
4. Keep API tokens in a local `.env` file
5. Run preprocessing and forecasting scripts from the repository root

## Notes
- Large raw data files and generated processed outputs are not committed to GitHub
- API tokens and local environment files must remain private and should not be pushed to GitHub