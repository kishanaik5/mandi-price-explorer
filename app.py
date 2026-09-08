"""Mandi Price Explorer — Streamlit UI.

Browse variety-wise daily Indian market prices from data.gov.in and compare modal
prices across markets. All business logic lives in ``services/``; this file is the
UI shell only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.datagov_client import fetch_records, unique_values
from utils.config import API_KEY_ENV, get_api_key

st.set_page_config(page_title="Mandi Price Explorer", page_icon="🌾", layout="wide")

st.title("🌾 Mandi Price Explorer")
st.caption(
    "Browse variety-wise daily Indian market prices and compare prices across markets — "
    "powered by the data.gov.in public API."
)


@st.cache_data(ttl=3600, show_spinner=False)
def _recent_states(api_key: str) -> list[str]:
    """Derive the State dropdown from recent records so spellings match the API."""
    df = fetch_records(api_key, filters=None, max_records=500, sort_desc=True)
    return unique_values(df, "State")


# ----------------------------- Sidebar: inputs ------------------------------
with st.sidebar:
    st.header("Settings")

    ui_key = st.text_input(
        "data.gov.in API key",
        type="password",
        help=f"Optional if {API_KEY_ENV} is set as a Space secret. Never stored.",
    )
    api_key = get_api_key(ui_key)

    if not api_key:
        st.warning(
            f"Add your data.gov.in API key above, or set the {API_KEY_ENV} "
            "secret. Get a free key at data.gov.in."
        )
        st.stop()

    st.divider()
    st.subheader("Filters")

    try:
        states = _recent_states(api_key)
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    if not states:
        st.error("No data returned from the API. Check your key or try again later.")
        st.stop()

    state = st.selectbox("State", options=states, index=0)
    max_records = st.slider(
        "Max records to fetch", min_value=200, max_value=5000, value=2000, step=200,
        help="Recent records for the chosen state (sorted by latest arrival date).",
    )

# --------------------------- Fetch state-level data --------------------------
try:
    with st.spinner(f"Fetching latest prices for {state}…"):
        data = fetch_records(api_key, filters={"State": state}, max_records=max_records)
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

if data.empty:
    st.info(f"No records found for {state}. Try another state or raise the record cap.")
    st.stop()

# ---------------- Sidebar: client-side cascading refinements -----------------
with st.sidebar:
    commodities = ["All"] + unique_values(data, "Commodity")
    commodity = st.selectbox("Commodity", options=commodities, index=0)

    scoped = data if commodity == "All" else data[data["Commodity"] == commodity]

    districts = ["All"] + unique_values(scoped, "District")
    district = st.selectbox("District", options=districts, index=0)
    if district != "All":
        scoped = scoped[scoped["District"] == district]

    varieties = ["All"] + unique_values(scoped, "Variety")
    variety = st.selectbox("Variety", options=varieties, index=0)
    if variety != "All":
        scoped = scoped[scoped["Variety"] == variety]

# ------------------------------- Main: results -------------------------------
latest_date = scoped["Arrival_Date"].max()
st.subheader(f"{commodity if commodity != 'All' else 'All commodities'} in {state}")
if pd.notna(latest_date):
    st.caption(f"Latest arrival date in view: {latest_date:%d %b %Y} · {len(scoped)} records")

# Summary metric cards
modal = scoped["Modal_Price"].dropna()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Markets", scoped["Market"].nunique())
c2.metric("Avg modal ₹/qtl", f"{modal.mean():,.0f}" if not modal.empty else "—")
c3.metric("Min modal ₹/qtl", f"{modal.min():,.0f}" if not modal.empty else "—")
c4.metric("Max modal ₹/qtl", f"{modal.max():,.0f}" if not modal.empty else "—")

# Price table
st.markdown("#### Price table")
table_cols = ["Arrival_Date", "Market", "District", "Commodity", "Variety", "Grade",
              "Min_Price", "Max_Price", "Modal_Price"]
table_cols = [c for c in table_cols if c in scoped.columns]
st.dataframe(
    scoped[table_cols].sort_values("Arrival_Date", ascending=False),
    use_container_width=True, hide_index=True,
)

# Bar chart: latest modal price per market
st.markdown("#### Modal price by market")
latest_per_market = (
    scoped.sort_values("Arrival_Date")
    .dropna(subset=["Modal_Price"])
    .groupby("Market", as_index=True)["Modal_Price"].last()
    .sort_values(ascending=False)
)
if not latest_per_market.empty:
    st.bar_chart(latest_per_market, height=350)
else:
    st.info("No modal prices available to chart for the current selection.")

# ------------------------------ Price forecast ------------------------------
st.divider()
st.markdown("#### 📈 Price forecast")
if commodity == "All":
    st.info("Select a specific commodity in the sidebar to forecast its modal price.")
else:
    fc_markets = unique_values(scoped, "Market")
    fcol1, fcol2 = st.columns([2, 1])
    fc_market = fcol1.selectbox("Market to forecast", options=fc_markets) if fc_markets else None
    horizon = fcol2.slider("Weeks ahead", 4, 12, 8)
    if fc_market and st.button("Run forecast"):
        from services.forecast import fetch_price_history, forecast_prices, to_weekly

        try:
            with st.spinner(f"Fetching history & forecasting {commodity} @ {fc_market}…"):
                hist_df = fetch_price_history(
                    api_key, commodity, state, market=fc_market,
                    variety=None if variety == "All" else variety,
                )
                result = forecast_prices(to_weekly(hist_df), horizon=horizon)
            chart = pd.concat(
                [result.history.rename("History"), result.forecast.rename("Forecast")], axis=1
            )
            st.line_chart(chart, height=320)
            m1, m2, m3 = st.columns(3)
            m1.metric("Weeks of history", result.n_weeks)
            m2.metric("Next-week forecast ₹/qtl", f"{result.forecast.iloc[0]:,.0f}")
            m3.metric("Backtest RMSE ₹", f"{result.rmse:,.0f}" if result.rmse else "—")
            st.caption(
                f"Trained on {result.training_window}. Recursive weekly forecast "
                "(each week feeds the next), smoothed to limit drift. Decision support only."
            )
        except (RuntimeError, ValueError) as exc:
            st.warning(str(exc))

# ------------------------------ How it works --------------------------------
with st.expander("ℹ️ How it works"):
    st.markdown(
        """
        **Pipeline:** data.gov.in API → filter → summarize → forecast

        1. **Fetch** — Records are pulled from the data.gov.in *Variety-wise Daily
           Market Prices* resource with pagination, sorted by `Arrival_Date`
           descending (the API defaults to the *oldest* rows, so we override it).
           Responses are cached for 1 hour (`@st.cache_data`).
        2. **Filter** — State is applied **server-side** (`filters[State]`); the
           State list itself is derived from recent records so spellings match the
           API exactly. Commodity, District, and Variety are refined client-side.
        3. **Summarize** — Min/Max/Modal prices feed the metric cards and table;
           the bar chart shows the latest modal price per market.
        4. **Forecast** — For a chosen commodity + market, the daily history is
           resampled to a weekly modal price, lag (1/2/7) + rolling + seasonal
           (sin/cos of week) features are built, a gradient-boosting model is fit,
           and a recursive multi-week forecast is rolled forward. A holdout
           backtest reports RMSE.

        **Note:** Forecasts are statistical decision-support, not guarantees.
        Prices are in ₹ per quintal as reported by the source.
        """
    )
