import numpy as np
import pandas as pd
from template.prelude_template import load_polymarket_data
from template.model_development_template import allocate_sequential_stable

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL = "CapMVRVCur"

def load_polymarket_sentiment() -> pd.DataFrame:
    # maps 'price' to 'probability
    data = load_polymarket_data()
    markets = data.get("markets", pd.DataFrame())
    odds = data.get("odds_history", pd.DataFrame())
    
    if markets.empty or odds.empty: 
        return pd.DataFrame()

    # Map ID columns dynamically
    m_id = "market_id" if "market_id" in markets.columns else "id"
    o_id = "market_id" if "market_id" in odds.columns else "id"
    
    # Filter for Macro/BTC Context
    btc_keywords = ['BTC', 'Bitcoin', 'Crypto', 'ETH', 'Solana', 'Fed', 'SEC', 'ETF']
    mask = markets["question"].str.contains('|'.join(btc_keywords), case=False, na=False)
    target_ids = set(markets.loc[mask, m_id].tolist())

    # Process Odds using 'price' as probability
    relevant_odds = odds[odds[o_id].isin(target_ids)].copy()
    if relevant_odds.empty:
        return pd.DataFrame()

    # Ensure date exists and use 'price' for the sentiment calculation
    relevant_odds["date"] = pd.to_datetime(relevant_odds["timestamp"]).dt.normalize()
    print("Revelent odds columns", relevant_odds.columns)
    # Use 'price' column since 'probability' is missing
    prob_col = "price" if "price" in relevant_odds.columns else "probability"
    
    daily_odds = relevant_odds.groupby("date")[prob_col].mean().rename("poly_sent")
    
    # Momentum Calculation
    sentiment = daily_odds.to_frame()
    sentiment["poly_mom"] = sentiment["poly_sent"].diff(7).fillna(0)
    sentiment["final_sent"] = (sentiment["poly_sent"] * 0.7) + (sentiment["poly_mom"] * 0.3)
    
    return sentiment[["final_sent"]]

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    price = df[PRICE_COL].copy()
    mvrv = df[MVRV_COL].copy()
    
    # Responsive MVRV (1-year context is better for the modern ETF era)
    rolling_mean = mvrv.rolling(365).mean()
    rolling_std = mvrv.rolling(365).std()
    mvrv_z = ((mvrv - rolling_mean) / rolling_std).clip(-3, 3).fillna(0)
    
    # Trend (200-day MA)
    ma200 = price.rolling(200).mean()
    price_vs_ma = ((price / ma200) - 1).clip(-1, 1).fillna(0)
    
    # Polymarket (Shifted and Reindexed)
    poly = load_polymarket_sentiment().reindex(price.index, method='ffill').fillna(0.5)
    
    features = pd.DataFrame({
        PRICE_COL: price,
        "mvrv_z": mvrv_z,
        "price_vs_ma": price_vs_ma,
        "poly": poly["final_sent"]
    })
    
    return features.shift(1).fillna(0)

def compute_dynamic_multiplier(mvrv_z, price_vs_ma, poly):
    """
    ASMMYETRIC ALLOCATOR:
    Heavy on Value during dips, light on Hype during rallies.
    """
    # BASE LOGIC: 
    # Inverse MVRV (buy more when low)
    # Inverse Price/MA (buy more when below trend)
    # Direct Poly (buy more when sentiment is rising)
    
    # Bear/Neutral Signal (Z < 1.0)
    bear_signal = (-mvrv_z * 0.65) + (-price_vs_ma * 0.25) + (poly * 0.10)
    
    # Overheated Signal (Z >= 1.0)
    # In overheated markets, we aggressively dial back to avoid 'Hype-Buying'
    bull_signal = (-mvrv_z * 0.40) + (-price_vs_ma * 0.50) + (poly * 0.10)
    
    combined = np.where(mvrv_z < 1.0, bear_signal, bull_signal)
    
    # Lower Strength (4.5) to maintain higher win-rate consistency
    return np.exp(np.clip(combined * 4.5, -4, 8))

def compute_window_weights(features_df, start_date, end_date):
    df = features_df.loc[start_date:end_date]
    n = len(df)
    if n == 0: return pd.Series()
    
    multipliers = compute_dynamic_multiplier(
        df["mvrv_z"].values,
        df["price_vs_ma"].values,
        df["poly"].values
    )
    
    raw_weights = (np.ones(n) / n) * multipliers
    weights = allocate_sequential_stable(raw_weights, n)
    return pd.Series(weights, index=df.index)