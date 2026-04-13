# 📦 Multi-SKU Demand Forecasting Engine

> **Gaurav Bhatia** · MSc Data Science | GISMA University, Berlin  
> [LinkedIn](https://linkedin.com/in/gaurav-bhatia-5a5a83184) · [GitHub](https://github.com/gauravbhatia-bit)

## What this project does
Forecasts weekly demand for 12 product SKUs using **ARIMA**, **Facebook Prophet**, and **XGBoost** with an interactive Streamlit dashboard.

Inspired by real inventory challenges at **Bhatia Traders** where forecasting reduced excess stock by 10%.

## Dataset
Download `train.csv` from Kaggle:  
👉 https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

## Run in Google Colab (easiest)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/gauravbhatia-bit/demand-forecasting-engine/blob/main/P1_Demand_Forecasting_Gaurav.ipynb)

## Run locally
```bash
pip install -r requirements.txt
python data/prepare_data.py
streamlit run app/streamlit_app.py
```

## Tech Stack
Python · ARIMA · Prophet · XGBoost · Optuna · Streamlit · Plotly · Pandas

## Related Projects
- [Bhatia Traders Sales Analysis](https://github.com/gauravbhatia-bit/bhatia-traders-sales-analysis)
- [Inventory Alert System](https://github.com/gauravbhatia-bit/bhatia_traders_inventory_alert_system)
