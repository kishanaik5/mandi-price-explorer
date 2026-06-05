"""Client for the data.gov.in Variety-wise Daily Market Prices API.

Field names below were discovered from the live ``field`` metadata of resource
``35985678-0d79-46b4-9ed6-6f13308a1d24`` and are used verbatim:

    Arrival_Date, Commodity, Commodity_Code, District, Grade, Market,
    Max_Price, Min_Price, Modal_Price, State, Variety

Notes learned from the API:
- The default ordering returns the OLDEST rows (year 2009), so we always request
  ``sort[Arrival_Date]=desc`` to surface current prices.
- Prices arrive as strings; ``Arrival_Date`` is ``dd/mm/yyyy``. We coerce both.
- The dataset has ~79M rows, so callers must always filter and cap pagination.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd
import requests
import streamlit as st

from utils.config import BASE_URL

# Exact field keys returned by the API.
PRICE_COLS: List[str] = ["Min_Price", "Max_Price", "Modal_Price"]
DATE_COL: str = "Arrival_Date"

# Server-side filterable fields (keyword exact-match) -> filters[<Field>]=value
FILTER_FIELDS: List[str] = ["State", "District", "Commodity", "Variety"]

_PAGE_SIZE: int = 1000
_TIMEOUT: int = 30

# data.gov.in's WAF stalls the default "python-requests" User-Agent (the response
# body never arrives), so we send an explicit UA. Any non-default value works.
_HEADERS = {"User-Agent": "mandi-price-explorer/1.0", "Accept": "application/json"}


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce price columns to numeric and parse the arrival date."""
    if df.empty:
        return df
    for col in PRICE_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if DATE_COL in df.columns:
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%d/%m/%Y", errors="coerce")
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_records(
    api_key: str,
    filters: Optional[Dict[str, str]] = None,
    max_records: int = 2000,
    sort_desc: bool = True,
) -> pd.DataFrame:
    """Fetch records with pagination, returning a normalized DataFrame.

    Args:
        api_key: data.gov.in API key (used only as a request param / cache key).
        filters: mapping of field name -> value for server-side ``filters[..]``.
        max_records: hard cap on rows pulled across pages.
        sort_desc: sort by ``Arrival_Date`` descending to get recent prices.

    Returns:
        A normalized DataFrame (numeric prices, parsed dates). Empty on no data.

    Raises:
        RuntimeError: on network/HTTP failure, with a user-friendly message.
    """
    filters = filters or {}
    params: Dict[str, str] = {"api-key": api_key, "format": "json", "limit": str(_PAGE_SIZE)}
    if sort_desc:
        params["sort[Arrival_Date]"] = "desc"
    for field, value in filters.items():
        if value:
            params[f"filters[{field}]"] = value

    records: List[dict] = []
    offset = 0
    try:
        while len(records) < max_records:
            params["offset"] = str(offset)
            params["limit"] = str(min(_PAGE_SIZE, max_records - len(records)))
            resp = requests.get(BASE_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT)
            resp.raise_for_status()
            payload = resp.json()
            page = payload.get("records", [])
            if not page:
                break
            records.extend(page)
            offset += len(page)
            if len(page) < int(params["limit"]):
                break  # last page
    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"Could not reach data.gov.in: {exc}. Check your network and API key."
        ) from exc
    except ValueError as exc:  # JSON decode error
        raise RuntimeError(
            "data.gov.in returned an unexpected (non-JSON) response. "
            "Your API key may be invalid or rate-limited."
        ) from exc

    return _normalize(pd.DataFrame(records))


def unique_values(df: pd.DataFrame, column: str) -> List[str]:
    """Return sorted unique non-null string values of a column (for dropdowns)."""
    if df.empty or column not in df.columns:
        return []
    return sorted(str(v) for v in df[column].dropna().unique())
