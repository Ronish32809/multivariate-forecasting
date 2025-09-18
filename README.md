# Multivariate KPI Forecasting

This repository contains code and notebooks for forecasting university key performance indicators (KPIs) using **Random Forest** and **LSTM**.  
The analysis covers **enrollment, employees, graduates, budgets, and R&D expenditure** with annual data from **IPEDS (2010–2023)** and **NSF HERD**.

## What’s Included
- Data preprocessing pipeline (cleaning, interpolation, normalization)
- Random Forest and LSTM forecasting models
- Evaluation using MAE, RMSE, MAPE, and R²
- Forecasts applied to Boise State University, Eastern Oregon University, and UAB

## Usage
1. Open the Jupyter notebook in `/notebooks`.
2. Run the preprocessing and model training cells.
3. Review forecast plots and error metrics.

## Requirements
Install dependencies:
```bash
pip install -r requirements.txt
