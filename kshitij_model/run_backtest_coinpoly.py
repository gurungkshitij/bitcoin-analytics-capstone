import logging
import pandas as pd
from pathlib import Path
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from kshitij_model.coin_poly_model import precompute_features, compute_window_weights

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