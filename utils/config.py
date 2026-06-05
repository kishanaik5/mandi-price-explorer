"""Configuration & secret loading for Mandi Price Explorer.

Secrets are read from the environment (set as a Hugging Face Space secret) or
supplied at runtime via a password input in the UI. Nothing is ever written to
disk or logs by this module.
"""
from __future__ import annotations

import os
from typing import Optional

# data.gov.in "Variety-wise Daily Market Prices Data of Commodity"
RESOURCE_ID: str = "35985678-0d79-46b4-9ed6-6f13308a1d24"
BASE_URL: str = f"https://api.data.gov.in/resource/{RESOURCE_ID}"

# Name of the environment variable / HF Space secret holding the API key.
API_KEY_ENV: str = "DATA_GOV_API_KEY"


def get_api_key(ui_key: Optional[str] = None) -> Optional[str]:
    """Resolve the data.gov.in API key.

    Precedence: a key pasted in the UI wins, otherwise fall back to the
    ``DATA_GOV_API_KEY`` environment variable. Returns ``None`` when neither is
    set so the caller can show a friendly prompt instead of crashing.
    """
    if ui_key and ui_key.strip():
        return ui_key.strip()
    env_key = os.environ.get(API_KEY_ENV)
    return env_key.strip() if env_key else None


def require_api_key(ui_key: Optional[str] = None) -> str:
    """Return a usable API key or raise a clear, actionable error."""
    key = get_api_key(ui_key)
    if not key:
        raise RuntimeError(
            f"No data.gov.in API key found. Set the {API_KEY_ENV} environment "
            "variable (HF Space secret) or paste your key in the sidebar. "
            "Get a free key at https://data.gov.in/."
        )
    return key
