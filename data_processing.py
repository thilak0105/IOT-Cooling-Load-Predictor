import numpy as np
import pandas as pd
from config import DATA_DIR

def load_and_preprocess_data():
    """Loads CSV files and initializes the preprocessing pipeline."""
    train_raw = pd.read_csv(
        DATA_DIR / "train.csv",
        usecols=["building_id", "meter", "timestamp", "meter_reading"],
    )
    weather_raw = pd.read_csv(
        DATA_DIR / "weather_train.csv",
        usecols=["site_id", "timestamp", "air_temperature"],
    )
    meta_raw = pd.read_csv(
        DATA_DIR / "building_metadata.csv",
        usecols=["building_id", "site_id"],
    )

    required_train = {"building_id", "meter", "timestamp", "meter_reading"}
    required_weather = {"site_id", "timestamp", "air_temperature"}
    required_meta = {"building_id", "site_id"}

    assert required_train.issubset(train_raw.columns), "Missing columns in train.csv"
    assert required_weather.issubset(weather_raw.columns), "Missing columns in weather_train.csv"
    assert required_meta.issubset(meta_raw.columns), "Missing columns in building_metadata.csv"

    train_raw["timestamp"] = pd.to_datetime(train_raw["timestamp"])
    weather_raw["timestamp"] = pd.to_datetime(weather_raw["timestamp"])

    df = (
        train_raw.merge(meta_raw, on="building_id", how="left")
        .merge(weather_raw, on=["site_id", "timestamp"], how="left")
    )

    df = df[df["meter"] == 1].copy()  # Cooling load only
    df = df.sort_values(["site_id", "timestamp"]).reset_index(drop=True)
    df["air_temperature"] = df.groupby("site_id")["air_temperature"].transform(
        lambda s: s.interpolate(method="linear").bfill().ffill()
    )

    print("Using DATA_DIR:", DATA_DIR)
    print("Loaded cooling records:", f"{len(df):,}")
    print("Time range:", df["timestamp"].min(), "->", df["timestamp"].max())
    
    return engineer_features(df)

def engineer_features(df):
    """Adds cyclical time formats, lags, and rolling metrics parameters."""
    df["hour"] = df["timestamp"].dt.hour
    df["dayofweek"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    df["occupancy"] = 0
    df.loc[(df["is_weekend"] == 0) & (df["hour"].between(8, 18)), "occupancy"] = 1

    df["temp_sq"] = df["air_temperature"] ** 2
    df["temp_cube"] = df["air_temperature"] ** 3
    df["temp_x_occupancy"] = df["air_temperature"] * df["occupancy"]

    df = df.sort_values(["building_id", "timestamp"]).reset_index(drop=True)
    df["y_log"] = np.log1p(df["meter_reading"])

    g = df.groupby("building_id", sort=False)
    df["lag_1"] = g["y_log"].shift(1)
    df["lag_24"] = g["y_log"].shift(24)
    df["lag_168"] = g["y_log"].shift(168)

    df["roll_mean_24"] = g["y_log"].apply(lambda s: s.shift(1).rolling(24, min_periods=12).mean()).reset_index(level=0, drop=True)
    df["roll_std_24"] = g["y_log"].apply(lambda s: s.shift(1).rolling(24, min_periods=12).std()).reset_index(level=0, drop=True)
    df["roll_mean_168"] = g["y_log"].apply(lambda s: s.shift(1).rolling(168, min_periods=48).mean()).reset_index(level=0, drop=True)

    memory_cols = ["lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"]
    model_df = df.dropna(subset=memory_cols).copy()

    print("Rows after lag/rolling filters:", f"{len(model_df):,}")
    return split_data(model_df)

def split_data(model_df):
    """Splits data temporally and builds log encoded features cleanly."""
    model_df = model_df.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(model_df) * 0.8)
    train_df = model_df.iloc[:split_idx].copy()
    test_df = model_df.iloc[split_idx:].copy()

    # Honest target encoding from train only
    building_mean_train = train_df.groupby("building_id")["y_log"].mean()
    global_log_mean = train_df["y_log"].mean()

    train_df["building_mean_log"] = train_df["building_id"].map(building_mean_train).fillna(global_log_mean)
    test_df["building_mean_log"] = test_df["building_id"].map(building_mean_train).fillna(global_log_mean)

    features = [
        "air_temperature", "temp_sq", "temp_cube", "occupancy", "is_weekend",
        "hour_sin", "hour_cos", "month_sin", "month_cos", "temp_x_occupancy",
        "building_mean_log", "lag_1", "lag_24", "lag_168", "roll_mean_24", "roll_std_24", "roll_mean_168"
    ]

    return train_df, test_df, features, model_df
