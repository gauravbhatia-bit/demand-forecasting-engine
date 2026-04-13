"""Run: streamlit run app/streamlit_app.py"""
import sys, os; sys.path.insert(0,".")
import streamlit as st, pandas as pd, numpy as np
import plotly.graph_objects as go
from itertools import product
import warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.WARNING)
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from xgboost import XGBRegressor
from src.features import engineer_features, get_feature_cols, split_sku

st.set_page_config(page_title="Demand Forecasting", page_icon="📦", layout="wide")
st.title("📦 Multi-SKU Demand Forecasting Engine")
st.markdown("**Gaurav Bhatia** — MSc Data Science | GISMA University, Berlin")

@st.cache_data
def load():
    raw  = pd.read_csv("data/weekly_sales.csv", parse_dates=["date"])
    feat = engineer_features(raw)
    return raw, feat

try:
    raw, feat = load()
except:
    st.error("Run data/prepare_data.py first."); st.stop()

skus = sorted(feat["sku"].unique())
st.sidebar.header("Settings")
sku   = st.sidebar.selectbox("SKU", skus)
model = st.sidebar.selectbox("Model", ["ARIMA","Prophet","XGBoost"])
tw    = st.sidebar.slider("Test weeks", 4, 24, 12)

raw_s = raw[raw["sku"]==sku].sort_values("date")
tr, te = split_sku(feat, sku, tw)
actual = te["demand"].values
fc = get_feature_cols(feat)

def calc(a,p): return {"MAE":round(float(np.mean(np.abs(a-p))),2),"RMSE":round(float(np.sqrt(np.mean((a-p)**2))),2),"MAPE%":round(float(np.mean(np.abs((a-p)/np.maximum(a,1)))*100),2)}

with st.spinner(f"Running {model}..."):
    if model=="ARIMA":
        baic,bo=np.inf,(1,1,1)
        for p,d,q in product(range(3),range(2),range(3)):
            try:
                r=ARIMA(raw_s["demand"].iloc[:-tw],order=(p,d,q)).fit()
                if r.aic<baic: baic,bo=r.aic,(p,d,q)
            except: pass
        prd=np.maximum(0,ARIMA(raw_s["demand"].iloc[:-tw],order=bo).fit().forecast(tw).values)
    elif model=="Prophet":
        pdf=raw_s.iloc[:-tw][["date","demand"]].rename(columns={"date":"ds","demand":"y"})
        mp=Prophet(yearly_seasonality=True,weekly_seasonality=False); mp.fit(pdf)
        prd=np.maximum(0,mp.predict(mp.make_future_dataframe(tw,"W"))["yhat"].values[-tw:])
    else:
        mx=XGBRegressor(n_estimators=200,max_depth=5,learning_rate=0.05,random_state=42,verbosity=0)
        mx.fit(tr[fc],tr["demand"]); prd=np.maximum(0,mx.predict(te[fc]))
    met=calc(actual,prd)

c1,c2,c3=st.columns(3)
c1.metric("MAE",f"{met['MAE']:.1f}"); c2.metric("RMSE",f"{met['RMSE']:.1f}"); c3.metric("MAPE",f"{met['MAPE%']:.1f}%")
fig=go.Figure()
fig.add_trace(go.Scatter(x=list(tr["date"]),y=tr["demand"].tolist(),name="Train",line=dict(color="#2563EB")))
fig.add_trace(go.Scatter(x=list(te["date"]),y=actual.tolist(),name="Actual",line=dict(color="#16A34A",width=2)))
fig.add_trace(go.Scatter(x=list(te["date"]),y=prd.tolist(),name=f"{model} Forecast",line=dict(color="#DC2626",dash="dash",width=2)))
fig.update_layout(title=f"{sku} — {model}",xaxis_title="Date",yaxis_title="Demand",hovermode="x unified",height=430,template="plotly_white")
st.plotly_chart(fig,use_container_width=True)
st.divider()
st.markdown("**Gaurav Bhatia** · [LinkedIn](https://linkedin.com/in/gaurav-bhatia-5a5a83184)")
