"""Validate order records and assign fixed 07:00/19:00 provider windows."""

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "order_provider_id",
    "ordertime",
    "transaction_type",
    "order_type",
]


def load_orders(path: Path) -> pd.DataFrame:
    """Read only required columns; patient and encounter fields are never loaded."""
    try:
        orders = pd.read_csv(
            path,
            usecols=REQUIRED_COLUMNS,
            dtype={"order_provider_id": "string"},
            low_memory=False,
        )
    except ValueError as exc:
        raise ValueError(
            "Input CSV must contain: " + ", ".join(REQUIRED_COLUMNS)
        ) from exc
    return prepare_orders(orders)


def prepare_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """Clean raw orders without changing the meaning of observed transactions."""
    missing = set(REQUIRED_COLUMNS) - set(orders.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(sorted(missing)))
    orders = orders[REQUIRED_COLUMNS].copy()
    try:
        orders["ordertime"] = pd.to_datetime(
            orders["ordertime"], errors="coerce", format="mixed"
        )
    except ValueError as exc:
        raise ValueError(
            "Use timestamps with a consistent timezone or no timezone."
        ) from exc
    if not pd.api.types.is_datetime64_any_dtype(orders["ordertime"]):
        raise ValueError("Use timestamps with a consistent timezone or no timezone.")
    providers = orders["order_provider_id"]
    orders = orders.loc[
        providers.notna()
        & providers.astype(str).str.strip().ne("")
        & orders["ordertime"].notna()
    ].copy()
    if orders.empty:
        raise ValueError("No usable orders remain after provider/timestamp validation.")

    orders["order_provider_id"] = orders["order_provider_id"].astype(str)
    orders["order_type"] = (
        orders["order_type"].astype("string").fillna("Unknown").astype(str)
    )
    tx = orders["transaction_type"].fillna("").astype(str).str.strip().str.upper()
    orders["is_change"] = (tx == "CHANGE").astype(int)
    orders["is_dc"] = (tx == "D/C").astype(int)
    orders["is_modified"] = ((orders["is_change"] + orders["is_dc"]) > 0).astype(int)

    hours = orders["ordertime"].dt.hour
    anchor = pd.Series(
        np.where((hours >= 7) & (hours < 19), "day", "night"), index=orders.index
    )
    shift_date = orders["ordertime"].dt.normalize() - pd.to_timedelta(
        (hours < 7).astype(int), unit="D"
    )
    orders["shift_id"] = (
        orders["order_provider_id"]
        + "_"
        + shift_date.dt.strftime("%Y-%m-%d")
        + "_"
        + anchor
    )
    return orders.sort_values(["order_provider_id", "ordertime"], kind="mergesort")
