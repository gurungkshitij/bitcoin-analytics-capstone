import logging
import pandas as pd
from pathlib import Path
from template.prelude_template import load_data
from template.backtest_template import run_full_analysis
from kshitij_model.model_development import precompute_features, compute_window_weights

def weight_wrapper(df_window: pd.DataFrame) -> pd.Series:
    # Integrate the Backtest Engine to my model
    global _FEATS
    return compute_window_weights(_FEATS, df_window.index.min(), df_window.index.max())

def main():
    global _FEATS
    logging.basicConfig(level=logging.INFO)
    
    # Load raw BTC data
    btc_df = load_data()
    
    # Build features
    _FEATS = precompute_features(btc_df)
    
    # Run the analysis (This handles the 2,554 windows)
    output_dir = Path(__file__).parent / "output"
    run_full_analysis(
        btc_df=btc_df,
        features_df=_FEATS,
        compute_weights_fn=weight_wrapper,
        output_dir=output_dir,
        strategy_label="Kshitij Custom DCA"
    )

if __name__ == "__main__":
    main()