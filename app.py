"""Mandi Price Explorer — Streamlit UI.

Browse variety-wise daily Indian market prices from data.gov.in, find the
nearest mandi, and compare modal prices across markets. All business logic lives
in ``services/``; this file is the UI shell only.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from services.datagov_client import fetch_records, unique_values
from services.geo import geocode_place, nearest_markets
from utils.config import API_KEY_ENV, get_api_key

st.set_page_config(page_title="Mandi Price Explorer", page_icon="🌾", layout="wide")

st.title("🌾 Mandi Price Explorer")
st.caption(
    "Browse variety-wise daily Indian market prices, find the nearest mandi, "
    "and compare prices across markets — powered by the data.gov.in public API."
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

    st.divider()
    st.subheader("Nearest mandi")
    place = st.text_input("Your city / district", placeholder="e.g. Pune")
    use_live_geo = st.checkbox(
        "Use live geocoder for misses", value=False,
        help="Falls back to OpenStreetMap (geopy) when a district isn't in the bundled lookup.",
    )

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

# Nearest mandi + map
st.markdown("#### Nearest mandis & map")
coords = geocode_place(place, use_live=use_live_geo) if place else None
if place and coords is None:
    st.warning(
        f"Couldn't locate '{place}' in the bundled lookup. Try a major district "
        "name or enable the live geocoder in the sidebar."
    )

if coords:
    user_lat, user_lon = coords
    ranked = nearest_markets(user_lat, user_lon, scoped, use_live=use_live_geo, top_n=10)
    if ranked.empty:
        st.info("None of the markets in view could be geocoded for ranking.")
    else:
        st.write(f"Closest mandis to **{place}** ({user_lat:.3f}, {user_lon:.3f}):")
        st.dataframe(ranked, use_container_width=True, hide_index=True)
        map_df = ranked[["lat", "lon"]].copy()
        map_df = pd.concat(
            [map_df, pd.DataFrame({"lat": [user_lat], "lon": [user_lon]})], ignore_index=True
        )
        st.map(map_df)
else:
    # Fall back to mapping all geocodable markets in view.
    from services.geo import attach_coords

    located = attach_coords(scoped, use_live=use_live_geo)
    if not located.empty:
        st.caption("Markets located from the bundled district lookup:")
        st.map(located[["lat", "lon"]])
    else:
        st.info("Enter your city/district above to rank the nearest mandis.")

# ------------------------------ How it works --------------------------------
with st.expander("ℹ️ How it works"):
    st.markdown(
        """
        **Pipeline:** data.gov.in API → filter → geo → visualize

        1. **Fetch** — Records are pulled from the data.gov.in *Variety-wise Daily
           Market Prices* resource with pagination, sorted by `Arrival_Date`
           descending (the API defaults to the *oldest* rows, so we override it).
           Responses are cached for 1 hour (`@st.cache_data`).
        2. **Filter** — State is applied **server-side** (`filters[State]`); the
           State list itself is derived from recent records so spellings match the
           API exactly. Commodity, District, and Variety are refined client-side.
        3. **Summarize** — Min/Max/Modal prices feed the metric cards and table;
           the bar chart shows the latest modal price per market.
        4. **Geo** — Markets carry no coordinates, so each is geocoded by its
           **District** against a bundled static lookup of major Indian districts
           (optional live OpenStreetMap fallback). Distances use the Haversine
           great-circle formula; the map shows you plus the nearest mandis.

        **Limitation:** the static lookup covers major districts only — markets in
        uncovered districts are omitted from ranking/map unless the live geocoder
        is enabled. Prices are in ₹ per quintal as reported by the source.
        """
    )
