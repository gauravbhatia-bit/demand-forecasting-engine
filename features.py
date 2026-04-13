"""Run: python data/prepare_data.py  (after placing train.csv in data/)"""
import pandas as pd, os

def prepare(input="data/train.csv", output="data/weekly_sales.csv", top_n=12):
    df = pd.read_csv(input, parse_dates=["date"])
    df["week_start"] = df["date"].dt.to_period("W-MON").apply(lambda r: r.start_time)
    weekly = (df.groupby(["week_start","store","item"])["sales"].sum()
               .reset_index().rename(columns={"week_start":"date","sales":"demand"}))
    weekly = weekly[weekly["store"]==1].copy()
    top_items = weekly.groupby("item")["demand"].sum().nlargest(top_n).index.tolist()
    weekly = weekly[weekly["item"].isin(top_items)].copy()
    item_map = {item: f"SKU_{str(i+1).zfill(2)}_Item_{item}" for i,item in enumerate(sorted(top_items))}
    weekly["sku"] = weekly["item"].map(item_map)
    weekly = weekly[["date","sku","demand"]].sort_values(["sku","date"]).reset_index(drop=True)
    weekly.to_csv(output, index=False)
    print(f"Saved {output} — {weekly.shape}")

if __name__ == "__main__":
    if not os.path.exists("data/train.csv"):
        print("Put train.csv from Kaggle inside data/ first!")
    else:
        prepare()
