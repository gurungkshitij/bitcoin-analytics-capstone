import logging
import numpy as np
import pandas as pd
from template.prelude_template import load_polymarket_data
from template.model_development_template import (
    allocate_sequential_stable,
    _clean_array,
)

PRICE_COL = "PriceUSD_coinmetrics"
MVRV_COL  = "CapMVRVCur"


EXP_STRENGTH = {
    "capitulation": 40,  
    "accumulation": 8.0,
    "bull_trend":   3.0,
    "euphoria":     0.01,  
}

REGIME_CEILING = {
    "capitulation": 60.0,  # Higher ceiling for the deepest bottoms
    "accumulation": 8.0,
    "bull_trend":   5.0,
    "euphoria":     1.0,
}

REGIME_WEIGHTS = {
    "capitulation": (0.50, 0.15, 0.30, 0.05), # 50% MVRV, 30% Supply Shock
    "accumulation": (0.40, 0.30, 0.20, 0.10), # 30% Demand during sideways
    "bull_trend":   (0.20, 0.40, 0.30, 0.10), # Demand leads the bull run
    "euphoria":     (0.80, 0.05, 0.05, 0.10), 
}

def _rolling_zscore(series, window):
    return ((series - series.rolling(window).mean()) / series.rolling(window).std()).fillna(0).clip(-3, 3)

def precompute_features(df: pd.DataFrame) -> pd.DataFrame:
    # Network Demand (Active Addresses)
    addr_mom = (df["AdrActCnt"] / df["AdrActCnt"].rolling(30).mean()).fillna(1.0)
    
    # Value Anchor (MVRV)
    mvrv_z = _rolling_zscore(df[MVRV_COL], 365)
    
    # Supply Shock Composite (NetFlow + SplyEx)
    net_flow = (df["FlowInExNtv"] - df["FlowOutExNtv"])
    supply_shock = -_rolling_zscore(df["SplyExNtv"], 90) - _rolling_zscore(net_flow, 90)
    
    # The Halving Signal (The missing piece for 70%+)
    # Uses days since last halving to create a decay/growth cycle
    # For simplicity, we use a proxy for the 4-year cycle momentum
    halving_proxy = (df.index.dayofyear / 365.0).astype(float) 
    
    # Momentum Signal (ROI) - Required for regime logic
    roi = df[PRICE_COL].pct_change(365).fillna(0)

    # Regime Logic
    regime = pd.Series("accumulation", index=df.index)
    # Reverting to -1.1 to catch the full capitulation window
    regime[(mvrv_z < -1.1) & (roi < -0.05)] = "capitulation" 
    regime[(mvrv_z > 2.2)] = "euphoria"
    
    # Widening the Bull Trend to capture more of the momentum
    regime[(roi > 0.3) & (mvrv_z > 0.1)] = "bull_trend"

    features = pd.DataFrame({
        "mvrv_z": mvrv_z,
        "demand": addr_mom,
        "supply": supply_shock,
        "halving": halving_proxy,
        "regime": regime
    }).shift(1).fillna(method='bfill')
    
    return features

def compute_dynamic_multiplier(row):
    r = row['regime']
    s = EXP_STRENGTH[r]
    w = REGIME_WEIGHTS[r]
    
    # The 'Structural' Score calculation
    # We remove the bias of 'Neutral' and focus on relative distance
    score = (
        w[0] * (-row['mvrv_z']) + 
        w[1] * (row['demand'] - 1.0) +
        w[2] * row['supply'] +
        w[3] * (0.5 - row['halving']) # Cyclical component
    )
    
    return np.exp(np.clip(score * s, -5.0, REGIME_CEILING[r]))