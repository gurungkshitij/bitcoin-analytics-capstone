import logging
import numpy as np
import pandas as pd
from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)

# Constants
PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"
DYNAMIC_STRENGTH = 5.0 

def load_polymarket_sentiment() -> pd.DataFrame:
    # Processes Polymarket data for BTC sentiment
    polymarket_data = load_polymarket_data()
    if "markets" not in polymarket_data:
        return pd.DataFrame()
    
    df = polymarket_data["markets"]
    btc = df[df["question"].str.contains("Bitcoin|BTC", case=False, na=False)].copy()
    if btc.empty: return pd.DataFrame()
    
    btc["date"] = pd.to_datetime(btc["created_at"]).dt.normalize()
    daily = btc.groupby("date").agg(vol=("volume", "sum")).reset_index().set_index("date")
    
    # Normalize volume to 0-1 score
    daily["polymarket_sentiment"] = daily["vol"].rolling(30, min_periods=1).rank(pct=True)
    return daily[["polymarket_sentiment"]]

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Core Feature Engineering
    price = df[PRICE_COL].copy()
    mvrv = df[MVRV_COL].copy()
    
    # MVRV Z-Score
    mean = mvrv.rolling(365).mean()
    std = mvrv.rolling(365).std()
    mvrv_z = ((mvrv - mean) / std).clip(-4, 4).fillna(0)
    
    # 200-day MA
    ma200 = price.rolling(200).mean()
    price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)
    
    # Polymarket
    poly = load_polymarket_sentiment().reindex(price.index, fill_value=0.5)
    
    features = pd.DataFrame({
        PRICE_COL: price,
        "mvrv_zscore": mvrv_z,
        "price_vs_ma": price_vs_ma,
        "polymarket_sentiment": poly["polymarket_sentiment"]
    }).shift(1).fillna(0) # Lag 1 day to prevent cheating
    
    return features

def compute_dynamic_multiplier(mvrv_z, price_vs_ma, poly_sent):
    # Strategy Logic.
    # 60% MVRV (Value), 20% Trend (MA), 20% Sentiment (Polymarket)
    combined = (-mvrv_z * 0.60) + (-price_vs_ma * 0.20) + ((poly_sent - 0.5) * 0.20)
    
    # Apply strength and convert to multiplier
    return np.exp(np.clip(combined * DYNAMIC_STRENGTH, -5, 10))

def compute_window_weights(features_df, start_date, end_date):
    # Slices the 1.0 budget across the window.
    df = features_df.loc[start_date:end_date]
    n = len(df)
    if n == 0: return pd.Series()
    
    multipliers = compute_dynamic_multiplier(
        df["mvrv_zscore"].values,
        df["price_vs_ma"].values,
        df["polymarket_sentiment"].values
    )
    
    raw_weights = (np.ones(n) / n) * multipliers
    # Enforce 1.0 constraint using the template's stable allocator
    weights = allocate_sequential_stable(raw_weights, n)
    return pd.Series(weights, index=df.index)