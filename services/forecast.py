"""Weekly modal-price forecasting for a commodity at a market.

Clean-room, public-data forecaster built on the data.gov.in history: resample to
weekly modal price, engineer lag + seasonal features, fit a gradient-boosting
model, and roll a recursive multi-week forecast forward. A simple holdout
backtest reports RMSE so users can gauge reliability.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from services.datagov_client import fetch_records

# Feature sets — the full set needs lag7/rolling7, the short set works on tiny series.
FEATURES = ["lag1", "lag2", "lag7", "rolling_mean", "rolling_std",
            "month", "week", "sin_season", "cos_season"]
SHORT_FEATURES = ["lag1", "lag2", "rolling_mean", "rolling_std",
                  "month", "week", "sin_season", "cos_season"]
MIN_WEEKS_REQUIRED = 12
_SMOOTHING = 0.2  # blend each prediction with the last value to avoid runaway drift


@dataclass
class ForecastResult:
    """Historical weekly series + the forward forecast and backtest metric."""

    history: pd.Series        # weekly modal price (indexed by week)
    forecast: pd.Series       # predicted weekly modal price (future weeks)
    rmse: Optional[float]     # holdout backtest RMSE (None if too short to test)
    training_window: str
    n_weeks: int


def fetch_price_history(
    api_key: str, commodity: str, state: str, market: Optional[str] = None,
    variety: Optional[str] = None, max_records: int = 4000,
) -> pd.DataFrame:
    """Fetch a longer price history for one commodity (optionally market/variety)."""
    filters = {"State": state, "Commodity": commodity}
    if market:
        filters["Market"] = market
    if variety:
        filters["Variety"] = variety
    return fetch_records(api_key, filters=filters, max_records=max_records, sort_desc=True)


def to_weekly(df: pd.DataFrame) -> pd.Series:
    """Collapse records to a weekly mean modal-price series."""
    if df.empty or "Modal_Price" not in df.columns:
        return pd.Series(dtype=float)
    s = df.dropna(subset=["Arrival_Date", "Modal_Price"]).set_index("Arrival_Date")["Modal_Price"]
    return s.resample("W").mean().dropna()


def _make_features(weekly: pd.Series, short: bool) -> pd.DataFrame:
    """Build lag + seasonal features from a weekly price series."""
    df = weekly.to_frame(name="Price").copy()
    df["lag1"] = df["Price"].shift(1)
    df["lag2"] = df["Price"].shift(2)
    roll = 3 if short else 7
    if not short:
        df["lag7"] = df["Price"].shift(7)
    df["rolling_mean"] = df["Price"].rolling(roll).mean()
    df["rolling_std"] = df["Price"].rolling(roll).std().fillna(0)
    df["month"] = df.index.month
    df["week"] = df.index.isocalendar().week.astype(float)
    df["sin_season"] = np.sin(2 * np.pi * df["week"] / 52)
    df["cos_season"] = np.cos(2 * np.pi * df["week"] / 52)
    return df.dropna()


def _recursive_forecast(
    model: GradientBoostingRegressor, history: List[float],
    last_date: pd.Timestamp, horizon: int, features: List[str], short: bool,
) -> List[float]:
    """Predict ``horizon`` weeks ahead, feeding each prediction back as a lag."""
    preds: List[float] = []
    for step in range(horizon):
        lag1 = history[-1]
        lag2 = history[-2] if len(history) >= 2 else lag1
        window = 3 if short else 7
        rolling_mean = float(np.mean(history[-window:]))
        rolling_std = float(np.std(history[-window:])) or 1.0
        future = last_date + pd.Timedelta(weeks=step + 1)
        week = float(future.isocalendar().week)
        row = {
            "lag1": lag1, "lag2": lag2, "rolling_mean": rolling_mean,
            "rolling_std": rolling_std, "month": future.month, "week": week,
            "sin_season": np.sin(2 * np.pi * week / 52),
            "cos_season": np.cos(2 * np.pi * week / 52),
        }
        if not short:
            row["lag7"] = history[-7] if len(history) >= 7 else history[0]
        pred = float(model.predict(pd.DataFrame([row])[features])[0])
        pred = (1 - _SMOOTHING) * pred + _SMOOTHING * lag1  # damp runaway drift
        history.append(pred)
        preds.append(pred)
    return preds


def forecast_prices(weekly: pd.Series, horizon: int = 8) -> ForecastResult:
    """Fit a model on the weekly series and forecast ``horizon`` weeks ahead.

    Raises:
        ValueError: if the series is too short to model.
    """
    weekly = weekly.dropna()
    if len(weekly) < 6:
        raise ValueError(
            f"Need at least 6 weeks of price history to forecast (got {len(weekly)}). "
            "Pick a commodity/market with more data or raise the record cap."
        )
    short = len(weekly) < MIN_WEEKS_REQUIRED
    features = SHORT_FEATURES if short else FEATURES
    window = "limited history" if short else "full history"

    feat = _make_features(weekly, short)
    if len(feat) < 4:
        raise ValueError("Not enough usable points after feature engineering.")
    X, y = feat[features], feat["Price"]

    # Holdout backtest on the last ~20% (min 2 weeks) when we have room.
    rmse: Optional[float] = None
    if len(feat) >= 8:
        split = max(2, int(len(feat) * 0.2))
        Xtr, ytr = X.iloc[:-split], y.iloc[:-split]
        Xte, yte = X.iloc[-split:], y.iloc[-split:]
        bt = GradientBoostingRegressor(random_state=0).fit(Xtr, ytr)
        rmse = float(np.sqrt(mean_squared_error(yte, bt.predict(Xte))))

    model = GradientBoostingRegressor(random_state=0).fit(X, y)
    history = list(weekly.values.astype(float))
    preds = _recursive_forecast(model, history, weekly.index[-1], horizon, features, short)
    idx = pd.date_range(weekly.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")
    return ForecastResult(
        history=weekly, forecast=pd.Series(preds, index=idx, name="forecast"),
        rmse=rmse, training_window=window, n_weeks=len(weekly),
    )
