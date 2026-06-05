---
title: Mandi Price Explorer
emoji: 🌾
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.40.2
app_file: app.py
pinned: false
---

# 🌾 Mandi Price Explorer

Browse **variety-wise daily Indian market (mandi) prices**, filter by
state/district/commodity/variety, find the **nearest mandi** to a location, and
compare modal prices across markets — built entirely on the public
[data.gov.in](https://data.gov.in/) API.

> **Independent, public-data build.** This is a clean-room reimplementation using
> only the public data.gov.in API and first principles. It does not reuse,
> import, or reproduce any private/company code or data.

## What it does

- Pulls live daily mandi prices (min / max / modal ₹ per quintal) from data.gov.in.
- Cascading sidebar filters: State (server-side) → Commodity → District → Variety.
- Summary metric cards, a sortable price table, and a modal-price bar chart.
- Nearest-mandi ranking by Haversine distance, plus an interactive map.

## Architecture (pipeline)

1. **Fetch** — Paginated requests to the *Variety-wise Daily Market Prices*
   resource (`35985678-0d79-46b4-9ed6-6f13308a1d24`), sorted by `Arrival_Date`
   descending (the API defaults to oldest rows), cached 1h with `@st.cache_data`.
2. **Filter** — `State` applied server-side via `filters[State]`; the state list
   is derived from recent records so spellings match the source. Commodity,
   District and Variety are refined client-side in pandas.
3. **Summarize** — Numeric coercion of price strings; metric cards, table, and a
   latest-modal-price-per-market bar chart.
4. **Geo** — Markets are geocoded by their `District` against a bundled static
   lookup of major Indian districts (optional live OpenStreetMap fallback via
   geopy); Haversine distances rank the nearest mandis and feed `st.map`.

## Public data source

- **data.gov.in — Variety-wise Daily Market Prices Data of Commodity**
  Endpoint: `https://api.data.gov.in/resource/35985678-0d79-46b4-9ed6-6f13308a1d24`
  Params: `?api-key=<KEY>&format=json&limit=<N>&offset=<M>` with optional
  `filters[State]=`, `sort[Arrival_Date]=desc`.
  Real field keys: `State, District, Market, Commodity, Variety, Grade,
  Arrival_Date, Min_Price, Max_Price, Modal_Price, Commodity_Code`.

## Geocoding limitation

Mandi records carry no coordinates. We map each market to its district's
coordinates using a **bundled static lookup of ~100 major districts**, so markets
in districts outside that table are excluded from the nearest-mandi ranking and
map. Enable the **live geocoder** checkbox to fall back to OpenStreetMap
(rate-limited, best-effort) for uncovered districts.

## Run locally

```bash
pip install -r requirements.txt
export DATA_GOV_API_KEY="your_key_here"   # or paste it in the sidebar
streamlit run app.py
```

Get a free API key at [data.gov.in](https://data.gov.in/). You can also paste the
key directly into the sidebar at runtime — it is never written to disk or logs.

## Set the secret on Hugging Face

In your Space: **Settings → Variables and secrets → New secret**

- Name: `DATA_GOV_API_KEY`
- Value: your data.gov.in API key

The app reads it via `os.environ`. Users can also paste their own key in the
sidebar if the secret is not set.
