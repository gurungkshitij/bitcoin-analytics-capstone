import numpy as np
import pandas as pd
import logging
from pathlib import Path
from template.prelude_template import load_polymarket_data, load_data
from template.model_development_template import allocate_sequential_stable, _clean_array
from template.backtest_template import run_full_analysis

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL  = "CapMVRVCur"

# --- THE "STABLE 70" PARAMETERS ---
EXP_STRENGTH = {
    "capitulation": 42.0,  
    "accumulation": 10.0,  
    "bull_trend":   3.5,
    "euphoria":     0.01,
}

REGIME_WEIGHTS = {
    "capitulation": (0.40, 0.15, 0.35, 0.05, 0.05),
    "accumulation": (0.30, 0.25, 0.30, 0.10, 0.05),
    "bull_trend":   (0.20, 0.30, 0.30, 0.10, 0.10),
    "euphoria":     (0.60, 0.10, 0.10, 0.10, 0.10),
}

REGIME_CEILING = {
    "capitulation": 70.0, 
    "accumulation": 10.0,
    "bull_trend":   5.0,
    "euphoria":     1.0,
}

# --- LOGIC FUNCTIONS ---

def load_polymarket_sentiment() -> pd.DataFrame:
    data = load_polymarket_data()
    markets = data.get("markets", pd.DataFrame())
    odds = data.get("odds_history", pd.DataFrame())
    
    if markets.empty or odds.empty: 
        return pd.DataFrame(columns=["poly"])

    m_id = "market_id" if "market_id" in markets.columns else "id"
    o_id = "market_id" if "market_id" in odds.columns else "id"
    prob_col = "price" if "price" in odds.columns else "probability"

    up_words = ['hit', 'above', 'high', 'up', 'reach', 'rise', 'ETF']
    up_mask = (markets["question"].str.contains('BTC|Bitcoin', case=False, na=False) & 
               markets["question"].str.contains('|'.join(up_words), case=False, na=False))
    
    bear_words = ['recession', 'inflation', 'layoffs', 'cut', 'unemployment', 'no jobs']
    bear_mask = markets["question"].str.contains('|'.join(bear_words), case=False, na=False)

    up_daily = pd.Series(dtype=float)
    up_ids = set(markets.loc[up_mask, m_id].tolist())
    if up_ids:
        up_odds = odds[odds[o_id].isin(up_ids)].copy()
        if not up_odds.empty:
            up_odds["date"] = pd.to_datetime(up_odds["timestamp"]).dt.normalize()
            up_daily = up_odds.groupby("date")[prob_col].mean()

    bear_daily = pd.Series(dtype=float)
    bear_ids = set(markets.loc[bear_mask, m_id].tolist())
    if bear_ids:
        bear_odds = odds[odds[o_id].isin(bear_ids)].copy()
        if not bear_odds.empty:
            bear_odds["date"] = pd.to_datetime(bear_odds["timestamp"]).dt.normalize()
            bear_daily = 1.0 - bear_odds.groupby("date")[prob_col].mean()

    combined = pd.concat([up_daily, bear_daily], axis=1).mean(axis=1)
    if combined.empty: return pd.DataFrame(columns=["poly"])
        
    return combined.rolling(3).mean().rename("poly").to_frame()

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    m_365 = df[MVRV_COL].rolling(365).mean()
    s_365 = df[MVRV_COL].rolling(365).std()
    mvrv_z = ((df[MVRV_COL] - m_365) / s_365).clip(-3, 3).fillna(0)
    
    addr_mom = (df["AdrActCnt"] / df["AdrActCnt"].rolling(30).mean()).fillna(1.0)
    
    net_flow = (df["FlowInExNtv"] - df["FlowOutExNtv"])
    supply_shock = (
        -((df["SplyExNtv"] - df["SplyExNtv"].rolling(90).mean()) / df["SplyExNtv"].rolling(90).std()) - \
        ((net_flow - net_flow.rolling(90).mean()) / net_flow.rolling(90).std())
    ).clip(-3, 3).fillna(0)
    
    sma_200 = df[PRICE_COL].rolling(200).mean()
    impulse = (df[PRICE_COL] / sma_200).fillna(1.0).clip(0.5, 2.0)
    vol_30 = (df[PRICE_COL].pct_change().rolling(30).std() * np.sqrt(365)).fillna(0)
    
    poly_data = load_polymarket_sentiment()
    poly = poly_data.reindex(df.index, method='ffill')
    
    roi = df["ROI1yr"].ffill()
    regime = pd.Series("accumulation", index=df.index)
    regime[(mvrv_z < -1.1) & (roi < -0.05)] = "capitulation" 
    regime[(mvrv_z > 2.2)] = "euphoria"
    regime[(roi > 0.4) & (mvrv_z > 0.1)] = "bull_trend"

    return pd.DataFrame({
        "mvrv_z": mvrv_z, "demand": addr_mom, "supply": supply_shock,
        "impulse": impulse, "vol": vol_30, "poly": poly["poly"], "regime": regime,
        "halving": (df.index.dayofyear / 365.0)
    }).shift(1).bfill()

def compute_dynamic_multiplier(row):
    r = row['regime']
    s = EXP_STRENGTH[r]
    w = list(REGIME_WEIGHTS[r])
    
    if np.isnan(row['poly']):
        w[0] += w[4] 
        w[4] = 0.0
        current_poly = 0.5 
    else:
        current_poly = row['poly']

    sentiment_signal = 0.5 - current_poly
    
    score = (
        w[0] * (-row['mvrv_z']) + 
        w[1] * (row['demand'] - 1.0) +
        w[2] * row['supply'] + 
        w[3] * (0.5 - row['halving']) +
        w[4] * sentiment_signal
    )
    
    if r == "capitulation":
        score += (row['vol'] * 0.2) 

    if row['impulse'] > 1.05:
        return np.exp(np.clip(score * s, 0.0, REGIME_CEILING[r]))
    
    return np.exp(np.clip(score * s, -5.0, REGIME_CEILING[r]))

def compute_window_weights(features_df, start_date, end_date):
    df = features_df.loc[start_date:end_date]
    if len(df) == 0: return pd.Series()
    multipliers = _clean_array(df.apply(compute_dynamic_multiplier, axis=1).values)
    weights = allocate_sequential_stable((np.ones(len(df))/len(df)) * multipliers, len(df))
    return pd.Series(weights, index=df.index)

# --- BACKTEST RUNNER ---

def weight_wrapper(df_window: pd.DataFrame) -> pd.Series:
    global _FEATS
    return compute_window_weights(_FEATS, df_window.index.min(), df_window.index.max())

def main():
    global _FEATS
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
    
    logging.info("Starting Stable-70 Integrated Backtest...")
    btc_df = load_data()
    
    logging.info("Building Feature Set...")
    _FEATS = precompute_features(btc_df)
    
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    logging.info("Executing Strategy...")
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATS,
        compute_weights_fn=weight_wrapper,
        output_dir=output_dir,
        strategy_label="BTC Macro-Fear Predatory v8.1"
    )

if __name__ == "__main__":
    main()