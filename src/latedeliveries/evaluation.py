"""Business cost and probability threshold utilities."""

import numpy as np
import pandas as pd

VOUCHER_COST = 4
MISSED_LATE_COST = 15
THRESHOLDS = np.arange(5, 100, 5) / 100


def order_cost(y_true, flagged):
    """Calculate the total cost of voucher decisions."""
    y = np.asarray(y_true)
    flags = np.asarray(flagged, dtype=bool)

    if y.shape != flags.shape:
        raise ValueError("Inputs must have the same shape")

    return int(
        VOUCHER_COST * flags.sum() + MISSED_LATE_COST * ((y == 1) & ~flags).sum()
    )


def choose_threshold(y_true, probabilities):
    """Find the lowest-cost threshold using training predictions."""
    p = np.asarray(probabilities)
    rows = []

    for t in THRESHOLDS:
        flagged = p >= t
        rows.append(
            {
                "threshold": round(float(t), 2),
                "cost": order_cost(y_true, flagged),
                "flagged": int(flagged.sum()),
            }
        )

    table = pd.DataFrame(rows)
    winner = table.sort_values(["cost", "threshold"]).iloc[0]

    return float(winner["threshold"]), table
