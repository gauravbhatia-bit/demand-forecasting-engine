# 📦 Multi-SKU Demand Forecasting Engine

**Gaurav Bhatia** — MSc Data Science, AI & Digital Business | GISMA University, Berlin  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://linkedin.com/in/gaurav-bhatia-5a5a83184)
[![GitHub](https://img.shields.io/badge/GitHub-gauravbhatia--bit-black?logo=github)](https://github.com/gauravbhatia-bit)

---

## 🧠 What This Project Does

A production-style demand forecasting engine that compares **three machine learning and statistical models** — ARIMA, Facebook Prophet, and XGBoost — across **12 real product SKUs** using 5 years of weekly sales data.

The project is deployed as an **interactive Streamlit dashboard** where users can select any SKU, choose a model, and adjust the forecast horizon — seeing live MAE, RMSE, and MAPE metrics instantly.

> 💡 Inspired by real inventory challenges at **Bhatia Traders** (Chandigarh, India), where applying demand forecasting reduced excess stock by **10%**.

---

## 🎯 Business Problem

Businesses that cannot accurately forecast demand face two costly problems:
- **Overstocking** → capital tied up in unsold inventory
- **Stockouts** → lost sales and unhappy customers

This engine tackles both by forecasting weekly demand per SKU with multiple models and selecting the best performer — enabling smarter purchasing and replenishment decisions.

---

## 🏗️ Project Architecture

```
demand-forecasting-engine/
│
├── P1_Demand_Forecasting_Gaurav.ipynb   # Main notebook (run top to bottom)
├── app/
│   └── streamlit_app.py                 # Interactive dashboard
├── data/
│   └── weekly_sales.csv                 # Prepared weekly data (auto-generated)
├── models/
│   └── SKU_XX_xgboost.pkl              # Saved XGBoost models per SKU
├── reports/
│   └── model_comparison.csv            # Full results table
├── requirements.txt
└── README.md
```

---

## 🔬 Methodology

### Step 1 — Data Preparation
- Raw daily sales data aggregated to **weekly** granularity
- Filtered to **top 12 best-selling SKUs** from Store 1
- 5-year date range (2013–2017), ~260 weekly observations per SKU

### Step 2 — Feature Engineering (for XGBoost)
- **Lag features**: demand from 1, 2, 4, 8, 13, 26, 52 weeks ago
- **Rolling statistics**: mean & std deviation over 4, 8, 13, 26-week windows
- **Calendar features**: week of year, month, quarter, year
- Strict **time-based train/test split** (no data leakage)

### Step 3 — Models Compared

| Model | Approach | Strengths |
|-------|----------|-----------|
| **ARIMA** | Statistical time series | Interpretable, handles trends |
| **Facebook Prophet** | Decomposition-based | Handles seasonality & holidays automatically |
| **XGBoost + Optuna** | Gradient boosting + HPO | Best accuracy, uses engineered features |

### Step 4 — Evaluation
- **Test set**: last 12 weeks held out per SKU
- **Metrics**: MAE, RMSE, MAPE%
- **Hyperparameter tuning**: Optuna with TimeSeriesSplit cross-validation (20 trials)

---

## 📊 Results

> ⚠️ Results below are indicative. Run the notebook to generate your exact figures.

| Model | Avg MAE | Avg RMSE | Avg MAPE% |
|-------|---------|----------|-----------|
| ARIMA | ~71.44 | ~90.11 | ~15.49% |
| Prophet | ~29.34 | ~35.94| ~6.07% |
| **XGBoost** | **~32.96** | **~40.24** | **~6.61%** ✅ |

*Run Step 6 in the notebook to populate this table with real numbers and update here.*

---

## 🖥️ Streamlit Dashboard Features

- **SKU selector** — switch between any of the 12 product SKUs
- **Model selector** — compare ARIMA, Prophet, XGBoost side by side
- **Adjustable test horizon** — slider from 4 to 24 weeks
- **Live metrics** — MAE, RMSE, MAPE update instantly
- **Interactive Plotly chart** — hover to see exact forecast vs actual values

---

## 🛠️ Tech Stack

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| Forecasting | ARIMA (statsmodels), Facebook Prophet, XGBoost |
| Hyperparameter Tuning | Optuna |
| Feature Engineering | Pandas, NumPy |
| Visualisation | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Notebook | Google Colab / Jupyter |

---

## 🚀 How to Run

### Option 1 — Google Colab (Recommended)
1. Open `P1_Demand_Forecasting_Gaurav.ipynb` in Google Colab
2. Run cells top to bottom (Shift+Enter or Runtime → Run All)
3. Download `train.csv` from [Kaggle](https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data) and upload when prompted
4. Launch the Streamlit dashboard in the final cell — you'll get a public ngrok URL

### Option 2 — Run Locally
```bash
git clone https://github.com/gauravbhatia-bit/demand-forecasting-engine
cd demand-forecasting-engine
pip install -r requirements.txt
python data/prepare_data.py
streamlit run app/streamlit_app.py
```

### Dataset
Download `train.csv` from Kaggle:  
👉 https://www.kaggle.com/competitions/demand-forecasting-kernels-only/data

---

## 🔑 Key Learnings

- **XGBoost with lag features** consistently outperforms statistical models on this dataset because lag features capture autocorrelation better than ARIMA's manual parameter tuning
- **Optuna** saves significant time vs. manual grid search — finds optimal hyperparameters in ~20 trials
- **No data leakage** is critical in time series — all features are strictly shifted backwards before training
- **MAPE alone** can be misleading for low-demand SKUs — always report MAE and RMSE alongside it

---

## 📁 Related Projects

- [Bhatia Traders Sales Analysis](https://github.com/gauravbhatia-bit/bhatia-traders-sales-analysis) — EDA & inventory analysis using Python/Pandas
- [Inventory Alert System](https://github.com/gauravbhatia-bit/bhatia_traders_inventory_alert_system) — SQL-based real-time stock monitoring

---

## 👤 About Me

I'm **Gaurav Bhatia**, currently pursuing an MSc in Data Science, AI & Digital Business at GISMA University of Applied Sciences in Berlin, Germany.

Before moving into data science, I managed operations at **Bhatia Traders** — a sauces & condiments business in Chandigarh, India — where I applied Python and SQL to solve real inventory and supply chain challenges. That hands-on experience is what drives my interest in building practical, business-focused data tools.

I'm actively looking for **Data Analyst / Data Science internship opportunities in Berlin and Germany**.  
Feel free to reach out on [LinkedIn](https://linkedin.com/in/gaurav-bhatia-5a5a83184) or via email at gauravbhatia.gb6@gmail.com.

---

## 📄 License
MIT License — free to use, modify, and distribute with attribution.
