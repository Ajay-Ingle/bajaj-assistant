"""Loads/caches the customer CSV.

A POC-scale dataset (10k loan rows) fits comfortably in memory, so this is
a simple module-level cache populated on first access. A production
system would back this with a real database instead.
"""

import pandas as pd

from app.config import CUSTOMER_CSV_PATH

_customer_df: pd.DataFrame | None = None


def load_customer_data() -> pd.DataFrame:
    """Load the customer loan CSV once per process and cache it in memory."""
    global _customer_df
    if _customer_df is None:
        _customer_df = pd.read_csv(CUSTOMER_CSV_PATH)
    return _customer_df
