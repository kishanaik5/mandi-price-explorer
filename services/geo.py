"""Geocoding and nearest-mandi logic.

Markets in the data.gov.in feed are named places without coordinates, so we
geocode them by matching their ``District`` (then ``Market`` name) against a
bundled static lookup of major Indian districts. An optional live geocoder
(geopy/Nominatim) is used only if the library is installed AND the caller opts
in. Distances use the Haversine great-circle formula.

Limitation: the static table covers major districts only; markets in uncovered
districts are dropped from the nearest-mandi ranking and the map. See the README.
"""
from __future__ import annotations

import math
import os
from functools import lru_cache
from typing import Optional, Tuple

import pandas as pd

_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
_LOOKUP_PATH = os.path.join(_DATA_DIR, "mandi_geo_lookup.csv")

EARTH_RADIUS_KM: float = 6371.0


@lru_cache(maxsize=1)
def load_geo_lookup() -> pd.DataFrame:
    """Load and index the bundled static district -> lat/lon lookup."""
    df = pd.read_csv(_LOOKUP_PATH)
    df["district_key"] = df["district"].str.strip().str.lower()
    return df


def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometers between two points."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat, dlon = rlat2 - rlat1, rlon2 - rlon1
    a = math.sin(dlat / 2) ** 2 + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def _static_lookup(name: str) -> Optional[Tuple[float, float]]:
    """Look up coordinates for a place name in the static table (case-insensitive)."""
    if not name:
        return None
    lookup = load_geo_lookup()
    hit = lookup[lookup["district_key"] == name.strip().lower()]
    if not hit.empty:
        row = hit.iloc[0]
        return float(row["lat"]), float(row["lon"])
    return None


def _live_geocode(name: str) -> Optional[Tuple[float, float]]:
    """Best-effort live geocode via geopy/Nominatim; returns None if unavailable."""
    try:
        from geopy.geocoders import Nominatim  # type: ignore

        geocoder = Nominatim(user_agent="mandi-price-explorer")
        loc = geocoder.geocode(f"{name}, India", timeout=10)
        if loc:
            return float(loc.latitude), float(loc.longitude)
    except Exception:
        return None
    return None


def geocode_place(name: str, use_live: bool = False) -> Optional[Tuple[float, float]]:
    """Geocode a city/district name: static table first, optional live fallback.

    Args:
        name: place name (city or district).
        use_live: if True and the static lookup misses, try the live geocoder.

    Returns:
        ``(lat, lon)`` or ``None`` if the place could not be located.
    """
    coords = _static_lookup(name)
    if coords is None and use_live:
        coords = _live_geocode(name)
    return coords


def attach_coords(df: pd.DataFrame, use_live: bool = False) -> pd.DataFrame:
    """Add ``lat``/``lon`` columns by geocoding each row's District.

    Rows whose district cannot be located are dropped (with coordinates absent
    there is nothing to map or rank). Returns a copy.
    """
    if df.empty or "District" not in df.columns:
        return df.assign(lat=pd.NA, lon=pd.NA).iloc[0:0]

    unique_districts = df["District"].dropna().unique()
    coord_map = {d: geocode_place(str(d), use_live=use_live) for d in unique_districts}

    out = df.copy()
    out["lat"] = out["District"].map(lambda d: coord_map.get(d, (None, None))[0] if coord_map.get(d) else None)
    out["lon"] = out["District"].map(lambda d: coord_map.get(d, (None, None))[1] if coord_map.get(d) else None)
    return out.dropna(subset=["lat", "lon"]).copy()


def nearest_markets(
    user_lat: float,
    user_lon: float,
    df: pd.DataFrame,
    use_live: bool = False,
    top_n: int = 10,
) -> pd.DataFrame:
    """Rank markets in ``df`` by Haversine distance to the user's coordinates.

    Args:
        user_lat, user_lon: the user's location.
        df: records DataFrame (must contain ``District`` and ``Market``).
        use_live: pass-through to geocoding for uncovered districts.
        top_n: how many nearest markets to return.

    Returns:
        DataFrame with ``Market, District, lat, lon, distance_km`` sorted nearest
        first (one row per market). Empty if nothing could be geocoded.
    """
    located = attach_coords(df, use_live=use_live)
    if located.empty:
        return located

    located["distance_km"] = located.apply(
        lambda r: round(haversine(user_lat, user_lon, r["lat"], r["lon"]), 1), axis=1
    )
    cols = ["Market", "District", "lat", "lon", "distance_km"]
    cols = [c for c in cols if c in located.columns]
    ranked = located[cols].drop_duplicates(subset=["Market", "District"]).sort_values("distance_km")
    return ranked.head(top_n).reset_index(drop=True)
