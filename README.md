# Multivariate KPI Forecasting

This repository documents a forecasting framework for university key performance indicators (KPIs) using **Random Forest** and **LSTM**.  
The analysis covers **enrollment, employees, graduates, budgets, and R&D expenditure** with annual data from **IPEDS (2010–2023)** and the **NSF HERD** survey.

## Project Overview
- Focuses on multivariate forecasting of higher education KPIs.  
- Implements Random Forest for stable baseline predictions.  
- Uses LSTM to capture potential trend shifts.  
- Evaluation metrics include MAE, RMSE, MAPE, and R².  
- Framework tested on multiple universities (Boise State University, Eastern Oregon University, and the University of Alabama at Birmingham).  

## Data Sources
The project relies on publicly available institutional datasets:
- [IPEDS](https://nces.ed.gov/ipeds/datacenter/DataFiles.aspx?year=2023&sid=60195fbd-d0c2-461f-8781-0ff6749fe829&rtid=1) – Annual data on enrollment, employees, and graduates.  
- [NSF HERD Survey](https://ncsesdata.nsf.gov/profiles/site?method=rankingBySource&ds=HERD&o=n&s=a) – Research and development expenditure data.  

