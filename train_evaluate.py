import pandas as pd
import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

from config import SEED, XGB_FINAL_ESTIMATORS, XGB_CV_ESTIMATORS
from utils import register_result, evaluate_model, to_sorted_df

def run_models(train_df, test_df, features):
    """Runs primary and side models logging statistics successfully."""
    X_train = train_df[features]
    y_train_log = train_df["y_log"]
    X_test = test_df[features]
    y_test_log = test_df["y_log"]
    
    results = {}
    ts_results = {}

    print("Train shape:", X_train.shape, "| Test shape:", X_test.shape)
    
    # Baseline Model
    baseline_model = DummyRegressor(strategy="mean")
    baseline_model.fit(X_train, y_train_log)
    y_pred_baseline_log = baseline_model.predict(X_test)
    register_result(results, "Baseline", y_test_log, y_pred_baseline_log)
    print("Baseline done:\n", pd.Series(results["Baseline"]))

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train_log)
    y_pred_lr_log = lr_model.predict(X_test)
    register_result(results, "Linear Regression", y_test_log, y_pred_lr_log)
    print("Linear Regression done:\n", pd.Series(results["Linear Regression"]))

    # XGBoost
    xgb_model = XGBRegressor(
        n_estimators=XGB_FINAL_ESTIMATORS,
        learning_rate=0.03,
        max_depth=10,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.1,
        reg_lambda=2.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=SEED,
        n_jobs=-1,
    )
    xgb_model.fit(X_train, y_train_log, eval_set=[(X_test, y_test_log)], verbose=False)
    y_pred_xgb_log = xgb_model.predict(X_test)
    register_result(results, "XGBoost", y_test_log, y_pred_xgb_log)
    print("XGBoost done:\n", pd.Series(results["XGBoost"]))

    # Random Forest
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_split=20,
        min_samples_leaf=8,
        n_jobs=-1,
        random_state=SEED,
    )
    rf_model.fit(X_train, y_train_log)
    y_pred_rf_log = rf_model.predict(X_test)
    register_result(results, "Random Forest", y_test_log, y_pred_rf_log)
    print("Random Forest done:\n", pd.Series(results["Random Forest"]))

    # Time-Series Models
    y_pred_ts_24_log = X_test["lag_24"].values
    y_pred_ts_168_log = X_test["lag_168"].values
    register_result(ts_results, "TS_SeasonalNaive_24h", y_test_log, y_pred_ts_24_log)
    register_result(ts_results, "TS_SeasonalNaive_168h", y_test_log, y_pred_ts_168_log)

    ar_cols = ["lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]
    ar_model = LinearRegression()
    ar_model.fit(X_train[ar_cols], y_train_log)
    y_pred_ar_log = ar_model.predict(X_test[ar_cols])
    register_result(ts_results, "TS_AutoReg_LinearLags", y_test_log, y_pred_ar_log)

    y_pred_blend_log = (
        0.45 * X_test["lag_24"].values
        + 0.25 * X_test["lag_168"].values
        + 0.20 * X_test["roll_mean_24"].values
        + 0.10 * X_test["roll_mean_168"].values
    )
    register_result(ts_results, "TS_WeightedLagBlend", y_test_log, y_pred_blend_log)

    return results, ts_results

def run_cross_validation(model_df, features):
    """Executes robust structured Cross Validation."""
    cv_source = model_df.sort_values("timestamp").reset_index(drop=True)
    ts_cv = TimeSeriesSplit(n_splits=5)
    cv_rows = []

    for fold, (tr_idx, va_idx) in enumerate(ts_cv.split(cv_source), start=1):
        tr = cv_source.iloc[tr_idx].copy()
        va = cv_source.iloc[va_idx].copy()

        fold_mean = tr.groupby("building_id")["y_log"].mean()
        fold_global = tr["y_log"].mean()
        tr["building_mean_log"] = tr["building_id"].map(fold_mean).fillna(fold_global)
        va["building_mean_log"] = va["building_id"].map(fold_mean).fillna(fold_global)

        Xtr, ytr = tr[features], tr["y_log"]
        Xva, yva = va[features], va["y_log"]

        fold_model = XGBRegressor(
            n_estimators=XGB_CV_ESTIMATORS,
            learning_rate=0.03,
            max_depth=10,
            min_child_weight=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.1,
            reg_lambda=2.0,
            objective="reg:squarederror",
            eval_metric="rmse",
            random_state=SEED,
            n_jobs=-1,
        )
        fold_model.fit(Xtr, ytr, verbose=False)
        yva_pred_log = fold_model.predict(Xva)
        m = evaluate_model(yva, yva_pred_log)
        m["Fold"] = fold
        cv_rows.append(m)

    cv_df = pd.DataFrame(cv_rows).set_index("Fold")
    return cv_df

def verify_consistency(results, ts_results, main_df):
    """Perform integrity checks to preserve data structures."""
    required_models = {"Baseline", "Linear Regression", "XGBoost", "Random Forest"}
    missing = required_models - set(results.keys())
    assert not missing, f"Missing models in results: {missing}"
    assert list(results.keys()).count("Baseline") == 1, "Baseline appears more than once in results"
    
    overlap = set(results.keys()) & set(ts_results.keys())
    assert len(overlap) == 0, f"Model name overlap between results and ts_results: {overlap}"
    assert set(main_df.index) == set(results.keys()), "main_df is not a pure view of results"
    print("All consistency checks passed.")
