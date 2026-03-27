import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

PRIMARY_METRIC = "Log_R2"
CLIP_NONNEGATIVE = True  # single policy used by all models

def evaluate_model(y_true_log, y_pred_log, clip_nonnegative=CLIP_NONNEGATIVE):
    """Calculate R2, RMSE on logical and original scales safely."""
    y_pred_log = np.asarray(y_pred_log)

    log_r2 = r2_score(y_true_log, y_pred_log)
    log_rmse = float(np.sqrt(mean_squared_error(y_true_log, y_pred_log)))

    y_true_orig = np.expm1(y_true_log)
    y_pred_orig = np.expm1(y_pred_log)
    if clip_nonnegative:
        y_pred_orig = np.maximum(0, y_pred_orig)

    orig_r2 = r2_score(y_true_orig, y_pred_orig)
    orig_rmse = float(np.sqrt(mean_squared_error(y_true_orig, y_pred_orig)))
    orig_mae = float(mean_absolute_error(y_true_orig, y_pred_orig))

    return {
        "Log_R2": float(log_r2),
        "Log_RMSE": log_rmse,
        "Orig_R2": float(orig_r2),
        "Orig_RMSE": orig_rmse,
        "Orig_MAE": orig_mae,
    }

def register_result(store, model_name, y_true_log, y_pred_log):
    """Stores the specific evaluation instance accurately formatted in storage."""
    store[model_name] = evaluate_model(y_true_log, y_pred_log)

def to_sorted_df(store, primary_metric=PRIMARY_METRIC):
    """Render to structured Pandas dataframe securely."""
    out = pd.DataFrame(store).T
    return out.sort_values(primary_metric, ascending=False)
