import os
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as plt_sns
import pandas as pd

from data_processing import load_and_preprocess_data
from train_evaluate import run_models, run_cross_validation, verify_consistency
from utils import to_sorted_df, PRIMARY_METRIC

warnings.filterwarnings("ignore")
plt.style.use("ggplot")
plt_sns.set_palette("Set2")
pd.set_option("display.max_columns", 200)

try:
    import jinja2  # noqa: F401
    print("jinja2 is available: styled HTML export will include row highlighting.")
except Exception:
    print("jinja2 not installed: styled HTML will fallback to plain HTML table.")
    print("Install with: %pip install jinja2")

def main():
    print("Environment ready.")
    
    # 1. Data load and preprocessing
    train_df, test_df, features, model_df = load_and_preprocess_data()
    
    # 2. Run models
    results, ts_results = run_models(train_df, test_df, features)
    
    # 3. Validation / Metrics Formatting
    ts_df = to_sorted_df(ts_results)
    print("\nTime-series model summary (separate store):")
    print(ts_df[["Log_R2", "Orig_R2", "Orig_RMSE", "Orig_MAE"]])
    
    cv_df = run_cross_validation(model_df, features)
    print("\nCross-validation diagnostics:")
    print(cv_df[["Log_R2", "Orig_R2", "Orig_RMSE", "Orig_MAE"]])
    
    orig_r2_span = cv_df["Orig_R2"].max() - cv_df["Orig_R2"].min()
    if cv_df["Orig_R2"].min() < 0.10 or orig_r2_span > 0.50:
        print("\nNOTE: Original-scale R2 is unstable across folds.")
        print("Log-scale metrics remain more stable and should be emphasized.")
    
    main_df = to_sorted_df(results)
    print("\nMain model comparison:")
    print(main_df[["Log_R2", "Orig_R2", "Orig_RMSE", "Orig_MAE"]])
    
    main_df_sorted = main_df.sort_values(PRIMARY_METRIC, ascending=False).copy()
    best_model_name = main_df_sorted.index[0]
    print(f"\nBest model by {PRIMARY_METRIC}: {best_model_name}")

    verify_consistency(results, ts_results, main_df)

    # 4. Exports
    EXPORT_DIR = Path.cwd() / "paper_outputs"
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)

    main_df_sorted.to_csv(EXPORT_DIR / "main_model_comparison.csv", index=True)
    ts_df.to_csv(EXPORT_DIR / "time_series_model_comparison.csv", index=True)
    cv_df.to_csv(EXPORT_DIR / "cv_diagnostics.csv", index=True)

    styled_main_df = None
    try:
        styled_main_df = (
            main_df_sorted.style
            .format({"Log_R2": "{:.4f}", "Orig_R2": "{:.4f}", "Orig_RMSE": "{:.2f}", "Orig_MAE": "{:.2f}"})
            .apply(lambda row: ["background-color: #d5f4e6" if row.name == best_model_name else "" for _ in row], axis=1)
        )
    except Exception:
        pass

    if styled_main_df is not None:
        styled_main_df.to_html(EXPORT_DIR / "main_model_comparison_styled.html")
    else:
        main_df_sorted.to_html(EXPORT_DIR / "main_model_comparison_styled.html")

    plt.figure(figsize=(9, 5), dpi=250)
    plot_df = main_df_sorted.reset_index().rename(columns={"index": "Model"})
    colors = ["#2ecc71" if m == best_model_name else "#3498db" for m in plot_df["Model"]]
    plt.bar(plot_df["Model"], plot_df[PRIMARY_METRIC], color=colors, alpha=0.85)
    plt.title(f"Main Models Comparison ({PRIMARY_METRIC})")
    plt.ylabel(PRIMARY_METRIC)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(EXPORT_DIR / f"main_models_{PRIMARY_METRIC}.png")
    plt.close()

    print("Exports written to:", EXPORT_DIR.resolve())

if __name__ == "__main__":
    main()
