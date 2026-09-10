"""Weekly modal-price forecasting for a commodity at a market.

Clean-room, public-data forecaster built on the data.gov.in history: resample to
weekly modal price, engineer lag + seasonal features, fit a small ensemble of
regressors, and roll a recursive multi-week forecast forward. A simple holdout
backtest reports RMSE so users can gauge reliability.

Two entry points:
  - forecast_prices(...)      -> longer-horizon forecast (default 8 weeks) using
                                  the full weekly history you pass in.
  - fit_and_forecast(...)     -> short-window mode: trains on a smaller recent
                                  slice and forecasts a few weeks ahead. Useful
                                  for a "what does the near-term look like"
                                  view in the playground UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from services.datagov_client import fetch_records

# Feature sets — the full set needs lag7/rolling7, the short set works on tiny series.
FEATURES = ["lag1", "lag2", "lag7", "rolling_mean", "rolling_std",
            "month", "week", "sin_season", "cos_season"]
SHORT_FEATURES = ["lag1", "lag2", "rolling_mean", "rolling_std",
                   "month", "week", "sin_season", "cos_season"]

MIN_WEEKS_REQUIRED = 12       # below this, fall back to the short feature set
FIT_FORECAST_WEEKS = 4        # horizon for fit_and_forecast()
_SMOOTHING = 0.2              # blend each prediction with the last value to avoid runaway drift

# Small ensemble — kept lightweight (no LightGBM/CatBoost/XGBoost) so the
# playground has no extra native-library install/build burden.
def _base_models() -> List[tuple]:
    return [
        ("GBR", GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, random_state=42)),
        ("RF", RandomForestRegressor(n_estimators=200, random_state=42)),
    ]


@dataclass
class ForecastResult:
    """Historical weekly series + the forward forecast and backtest metric."""

    history: pd.Series                    # weekly modal price (indexed by week)
    forecast: pd.Series                   # ensemble-mean predicted weekly modal price
    model_forecasts: Dict[str, pd.Series] = field(default_factory=dict)  # per-model forecasts
    rmse: Optional[Dict[str, float]] = None  # per-model holdout RMSE (None if too short to test)
    training_window: str = ""
    n_weeks: int = 0


@dataclass
class FitForecastResult:
    """Short-window fit + near-term forecast, for a compact playground view."""

    train_dates: pd.Index
    train_actual: pd.Series
    train_fits: Dict[str, pd.Series]        # per-model in-sample fit
    forecast_dates: pd.Index
    forecast_preds: Dict[str, pd.Series]    # per-model forecast
    forecast_mean: pd.Series                # ensemble mean forecast
    mse: Dict[str, float]                   # per-model in-sample MSE
    from_date: pd.Timestamp
    to_date: pd.Timestamp


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
    model, history: List[float], last_date: pd.Timestamp,
    horizon: int, features: List[str], short: bool,
    scaler: Optional[StandardScaler] = None,
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
        X_row = pd.DataFrame([row])[features]
        if scaler is not None:
            X_row = pd.DataFrame(scaler.transform(X_row), columns=features)
        pred = float(model.predict(X_row)[0])
        pred = (1 - _SMOOTHING) * pred + _SMOOTHING * lag1  # damp runaway drift
        history.append(pred)
        preds.append(pred)
    return preds


def forecast_prices(weekly: pd.Series, horizon: int = 8) -> ForecastResult:
    """Fit an ensemble on the weekly series and forecast ``horizon`` weeks ahead.

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
    rmse: Dict[str, float] = {}
    have_backtest = len(feat) >= 8
    if have_backtest:
        split = max(2, int(len(feat) * 0.2))
        Xtr, ytr = X.iloc[:-split], y.iloc[:-split]
        Xte, yte = X.iloc[-split:], y.iloc[-split:]

    history_base = list(weekly.values.astype(float))
    model_forecasts: Dict[str, pd.Series] = {}
    idx = pd.date_range(weekly.index[-1] + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    for name, model in _base_models():
        if have_backtest:
            bt = type(model)(**model.get_params()).fit(Xtr, ytr)
            rmse[name] = float(np.sqrt(mean_squared_error(yte, bt.predict(Xte))))
        model.fit(X, y)
        preds = _recursive_forecast(model, list(history_base), weekly.index[-1], horizon, features, short)
        model_forecasts[name] = pd.Series(preds, index=idx, name=name)

    ensemble_mean = pd.Series(
        np.mean([s.values for s in model_forecasts.values()], axis=0), index=idx, name="forecast"
    )

    return ForecastResult(
        history=weekly,
        forecast=ensemble_mean,
        model_forecasts=model_forecasts,
        rmse=rmse or None,
        training_window=window,
        n_weeks=len(weekly),
    )


def fit_and_forecast(
    weekly: pd.Series,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: int = FIT_FORECAST_WEEKS,
) -> FitForecastResult:
    """Train on a (usually recent) slice of the weekly series and forecast a
    short horizon ahead. Always uses the short feature set, since this mode is
    meant for compact/recent windows rather than long history.

    Raises:
        ValueError: if the resulting window is too short to model.
    """
    weekly = weekly.dropna()
    if from_date:
        weekly = weekly[weekly.index >= pd.Timestamp(from_date)]
    if to_date:
        weekly = weekly[weekly.index <= pd.Timestamp(to_date)]

    feat = _make_features(weekly, short=True)
    if len(feat) < 3:
        raise ValueError(
            f"Not enough weekly points in this window to fit (got {len(feat)} after "
            "feature engineering, need at least 3). Widen the date range."
        )

    X_train = feat[SHORT_FEATURES]
    y_train = feat["Price"]
    scaler = StandardScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=SHORT_FEATURES, index=X_train.index)

    last_date = feat.index[-1]
    history_base = list(weekly.reindex(feat.index).values.astype(float))
    future_dates = pd.date_range(last_date + pd.Timedelta(weeks=1), periods=horizon, freq="W")

    mse: Dict[str, float] = {}
    train_fits: Dict[str, pd.Series] = {}
    forecast_preds: Dict[str, pd.Series] = {}

    for name, model in _base_models():
        model.fit(X_scaled, y_train)
        fit_vals = model.predict(X_scaled)
        mse[name] = round(float(mean_squared_error(y_train, fit_vals)), 2)
        train_fits[name] = pd.Series(fit_vals, index=feat.index, name=name)
        preds = _recursive_forecast(
            model, list(history_base), last_date, horizon, SHORT_FEATURES, short=True, scaler=scaler,
        )
        forecast_preds[name] = pd.Series(preds, index=future_dates, name=name)

    forecast_mean = pd.Series(
        np.mean([s.values for s in forecast_preds.values()], axis=0), index=future_dates, name="forecast",
    )

    return FitForecastResult(
        train_dates=feat.index,
        train_actual=feat["Price"],
        train_fits=train_fits,
        forecast_dates=future_dates,
        forecast_preds=forecast_preds,
        forecast_mean=forecast_mean,
        mse=mse,
        from_date=weekly.index[0],
        to_date=weekly.index[-1],
    )
