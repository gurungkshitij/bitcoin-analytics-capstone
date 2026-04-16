import logging
import pandas as pd
from pathlib import Path
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from kshitij_model.model_development_v2 import precompute_features, compute_dynamic_multiplier
from template.model_development_template import allocate_sequential_stable, _clean_array

def weight_wrapper(df_window: pd.DataFrame) -> pd.Series:
    """
    This bridges your window-based backtester to your daily multiplier logic.
    It takes a 365-day slice and applies the allocation logic.
    """
    global _FEATS
    
    # Slice features for this specific window
    window_feats = _FEATS.loc[df_window.index]
    
    # Generate Raw Multipliers
    # These use the predatory logic (EXP_STRENGTH 35.0, etc.)
    raw_mults = window_feats.apply(compute_dynamic_multiplier, axis=1).values
    raw_mults = _clean_array(raw_mults)
    
    # Allocation Logic
    n = len(raw_mults)
    base_weights = (pd.Series(1.0, index=df_window.index) / n) * raw_mults
    
    # Sequential Stable Allocation 
    final_weights = allocate_sequential_stable(base_weights.values, n)
    
    return pd.Series(final_weights, index=df_window.index)

def main():
    global _FEATS
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    
    # Load official BTC data 
    logging.info("Loading official data via load_data()...")
    btc_df = load_data()
    
    # Build your predatory features (v6.1 logic)
    logging.info("Precomputing predatory features (MVRV, Demand, Supply)...")
    _FEATS = precompute_features(btc_df)
    
    # Run the official analysis
    output_dir = Path(__file__).parent / "output"
    logging.info("Starting full analysis. This may take a minute...")
    
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATS,
        compute_weights_fn=weight_wrapper,
        output_dir=output_dir,
        strategy_label="BTC Predatory Accumulation v6.1"
    )

if __name__ == "__main__":
    main()