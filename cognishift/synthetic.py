"""Fictional order streams for exercising the pipeline, never performance evidence."""

import numpy as np
import pandas as pd


def make_synthetic_orders(seed: int = 42) -> pd.DataFrame:
    """Return repeatable, clearly fictional records with both outcome classes."""
    rng = np.random.default_rng(seed)
    rows = []
    for provider in range(24):
        for window in range(4):
            start = pd.Timestamp("2026-01-05") + pd.Timedelta(
                days=window, hours=8 if window % 2 == 0 else 20
            )
            activity = rng.choice([0.02, 0.12, 0.3, 0.65])
            for order in range(15):
                modified = rng.random() < activity
                rows.append(
                    {
                        "order_provider_id": f"synthetic_provider_{provider:03d}",
                        "ordertime": start + pd.Timedelta(minutes=order * 3),
                        "transaction_type": (
                            rng.choice(["Change", "D/C"]) if modified else "New"
                        ),
                        "order_type": rng.choice(["Medication", "Lab", "Imaging"]),
                    }
                )
    return pd.DataFrame(rows)
