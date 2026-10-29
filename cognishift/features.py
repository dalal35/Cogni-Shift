"""Construct temporally separated first-five predictors and next-ten outcomes."""

import math

import numpy as np
import pandas as pd

FIRST_N = 5
NEXT_N = 10

FEATURES = [
    "early_changes",
    "early_discontinuations",
    "early_modification_rate",
    "early_order_types",
    "early_order_type_entropy",
    "early_task_switch_rate",
    "early_median_gap_minutes",
    "early_orders_per_hour",
    "early_medication_share",
    "start_hour_sin",
    "start_hour_cos",
    "start_weekend",
]


def build_shift_examples(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for shift_id, group in orders.groupby("shift_id", sort=False):
        group = group.sort_values("ordertime", kind="mergesort")

        if len(group) < FIRST_N + NEXT_N:
            continue

        # Tied timestamps retain source-row order; they do not establish event precedence.
        early = group.iloc[:FIRST_N]
        future = group.iloc[FIRST_N : FIRST_N + NEXT_N]

        first_time = early["ordertime"].iloc[0]
        hours = first_time.hour + first_time.minute / 60

        types = early["order_type"]
        proportions = types.value_counts(normalize=True)

        entropy = -float(np.sum(proportions * np.log2(proportions)))

        gaps = early["ordertime"].diff().dt.total_seconds().dropna() / 60

        positive_gaps = gaps[gaps > 0]

        observed_hours = max(
            (early["ordertime"].iloc[-1] - first_time).total_seconds() / 3600,
            0.25,
        )

        rows.append(
            {
                "shift_id": shift_id,
                "provider_id": early["order_provider_id"].iloc[0],
                "shift_start": first_time,
                "future_modification_rate": float(future["is_modified"].mean()),
                "early_changes": int(early["is_change"].sum()),
                "early_discontinuations": int(early["is_dc"].sum()),
                "early_modification_rate": float(early["is_modified"].mean()),
                "early_order_types": int(types.nunique()),
                "early_order_type_entropy": entropy,
                "early_task_switch_rate": float(
                    (types != types.shift()).iloc[1:].mean()
                ),
                "early_median_gap_minutes": (
                    float(positive_gaps.median()) if len(positive_gaps) else np.nan
                ),
                "early_orders_per_hour": min(FIRST_N / observed_hours, 60.0),
                "early_medication_share": float(
                    types.str.contains("med", case=False, na=False).mean()
                ),
                "start_hour_sin": math.sin(2 * math.pi * hours / 24),
                "start_hour_cos": math.cos(2 * math.pi * hours / 24),
                "start_weekend": int(first_time.day_of_week >= 5),
            }
        )

    if not rows:
        raise ValueError(
            "No shifts have 15 or more orders. Check the CSV and shift rules."
        )

    return pd.DataFrame(rows)
