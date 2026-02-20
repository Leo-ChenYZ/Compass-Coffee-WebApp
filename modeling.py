"""
modeling.py
-----------
Fits a Random Forest on all available preprocessed data and predicts
total demand for the next 7 days per item.

Expected input (from preprocessing.py):
    A pd.DataFrame with columns:
        item, category, date, actual_qty_in_base_unit

Public API:
    run_forecast(df) -> pd.DataFrame with columns [Product, Predicted Amount (Next 7 Days)]
"""

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from pandas.tseries.holiday import USFederalHolidayCalendar


# ============================================================
# BEST PARAMS (from CV training)
# ============================================================

BEST_PARAMS = {
    'n_estimators': 600, 
    'max_depth': 12, 
    'min_samples_leaf': 1, 
    'max_features': 'sqrt'
}

CAT_COLS   = ["item", "category"]
NUM_COLS   = [
    "lag_1", "lag_7", "lag_14",
    "dow_sin", "dow_cos",
    "day_of_year_sin", "day_of_year_cos",
    "month_sin", "month_cos",
    "is_holiday",
]
TARGET_COL = "actual_qty_in_base_unit"
HORIZON    = 7


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def _add_features(df):
    """Add calendar, cyclical, lag, and holiday features to historical data."""
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["item", "date"])

    dow         = df["date"].dt.weekday
    month       = df["date"].dt.month
    day_of_year = df["date"].dt.dayofyear

    df["dow_sin"]         = np.sin(2 * np.pi * dow         / 7)
    df["dow_cos"]         = np.cos(2 * np.pi * dow         / 7)
    df["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    df["month_sin"]       = np.sin(2 * np.pi * month       / 12)
    df["month_cos"]       = np.cos(2 * np.pi * month       / 12)

    df["lag_1"]  = df.groupby("item")[TARGET_COL].shift(1)
    df["lag_7"]  = df.groupby("item")[TARGET_COL].shift(7)
    df["lag_14"] = df.groupby("item")[TARGET_COL].shift(14)

    cal      = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df["date"].min(), end=df["date"].max())
    df["is_holiday"] = df["date"].isin(holidays).astype(int)

    return df


def _build_future_rows(df):
    """
    Create one row per (item, forecast_date) for the next HORIZON days.
    Lags are filled from the tail of historical actuals (static — same lag
    value for all 7 forecast days, as actuals for future days are unknown).
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    last_date    = df["date"].max()
    all_items    = df["item"].unique()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D",
    )

    # Skeleton future grid
    future = pd.DataFrame(
        [(item, date) for item in all_items for date in future_dates],
        columns=["item", "date"],
    )

    # Carry forward category per item
    item_category = df.groupby("item")["category"].last().reset_index()
    future = future.merge(item_category, on="item", how="left")

    # Assign lag values from the tail of each item's history
    for item in all_items:
        mask      = future["item"] == item
        item_hist = df[df["item"] == item].sort_values("date")[TARGET_COL].values

        future.loc[mask, "lag_1"]  = item_hist[-1]  if len(item_hist) >= 1  else 0.0
        future.loc[mask, "lag_7"]  = item_hist[-7]  if len(item_hist) >= 7  else 0.0
        future.loc[mask, "lag_14"] = item_hist[-14] if len(item_hist) >= 14 else 0.0

    # Calendar features
    dow         = future["date"].dt.weekday
    month       = future["date"].dt.month
    day_of_year = future["date"].dt.dayofyear

    future["dow_sin"]         = np.sin(2 * np.pi * dow         / 7)
    future["dow_cos"]         = np.cos(2 * np.pi * dow         / 7)
    future["day_of_year_sin"] = np.sin(2 * np.pi * day_of_year / 365)
    future["day_of_year_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    future["month_sin"]       = np.sin(2 * np.pi * month       / 12)
    future["month_cos"]       = np.cos(2 * np.pi * month       / 12)

    # Holiday feature
    cal      = USFederalHolidayCalendar()
    holidays = cal.holidays(start=future["date"].min(), end=future["date"].max())
    future["is_holiday"] = future["date"].isin(holidays).astype(int)

    return future


# ============================================================
# MAIN ENTRY POINT
# ============================================================

def run_forecast(df):
    """
    Fit model on all available data, predict the next 7 days, and return
    a summary table of total predicted demand per item.

    Parameters
    ----------
    df : pd.DataFrame
        Preprocessed data from preprocessing.py with columns:
        item, category, date, actual_qty_in_base_unit

    Returns
    -------
    pd.DataFrame
        Columns: ['Product', 'Predicted Amount (Next 7 Days)']
        Sorted descending by predicted amount.
    """
    # 1. Build features on historical data
    model_df = _add_features(df)
    model_df = model_df.dropna(subset=NUM_COLS)  # remove rows with incomplete lags

    # 2. Fit TargetEncoder + RandomForest on all historical data
    te = TargetEncoder(cols=CAT_COLS)
    X  = te.fit_transform(model_df[CAT_COLS + NUM_COLS], model_df[TARGET_COL])
    y  = model_df[TARGET_COL].values

    rf = RandomForestRegressor(**BEST_PARAMS)
    rf.fit(X, y)

    # 3. Build future feature rows (next 7 days)
    future_df = _build_future_rows(model_df)

    # 4. Predict and clip negatives
    X_future          = te.transform(future_df[CAT_COLS + NUM_COLS])
    future_df["pred"] = np.maximum(rf.predict(X_future), 0.0)

    # 5. Sum across the 7-day horizon per item
    result = (
        future_df.groupby("item")["pred"]
                 .sum()
                 .round(1)
                 .reset_index()
                 .rename(columns={
                     "item": "Product",
                     "pred": "Predicted Amount (Next 7 Days)",
                 })
                 .sort_values("Predicted Amount (Next 7 Days)", ascending=False)
                 .reset_index(drop=True)
    )

    return result