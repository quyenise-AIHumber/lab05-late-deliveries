"""Train, evaluate, and save the late-delivery model."""

import json
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from latedeliveries.data import make_orders
from latedeliveries.evaluation import order_cost
from latedeliveries.model_selection import compare_and_tune
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    split_orders,
)


def main():
    """Rebuild reports, final model, and metadata."""

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # All model selection and threshold decisions
    # use training data only.
    pipeline, threshold, best_params, model_name = compare_and_tune()

    orders = make_orders()
    X_train, X_test, y_train, y_test = split_orders(orders)

    # Fit the chosen pipeline on ALL training rows.
    pipeline.fit(X_train, y_train)

    # FINAL TEST EVALUATION:
    # Predict on the untouched test set exactly once.
    probabilities = pipeline.predict_proba(X_test)[:, 1]

    predicted = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_test,
        predicted,
        labels=[0, 1],
    ).ravel()

    # Compare the three business policies.
    nobody = np.zeros(len(y_test), dtype=bool)
    everybody = np.ones(len(y_test), dtype=bool)
    model_flags = predicted.astype(bool)

    cost_nobody = order_cost(y_test, nobody)
    cost_everybody = order_cost(y_test, everybody)
    cost_model = order_cost(y_test, model_flags)

    metrics = {
        "test_samples": int(len(y_test)),
        "threshold": float(threshold),
        "confusion_matrix": [
            [int(tn), int(fp)],
            [int(fn), int(tp)],
        ],
        "precision": float(precision_score(y_test, predicted, zero_division=0)),
        "recall": float(recall_score(y_test, predicted, zero_division=0)),
        "f1": float(f1_score(y_test, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "costs_usd": {
            "flag_nobody": cost_nobody,
            "flag_everybody": cost_everybody,
            "model": cost_model,
        },
        "savings_vs_nobody_usd": (cost_nobody - cost_model),
    }

    # Save final test results.
    Path("reports/metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )

    # Save the WHOLE fitted pipeline.
    model_path = Path("models/late_orders.joblib")
    joblib.dump(pipeline, model_path)

    model_info = {
        "model_name": model_name,
        "scikit_learn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": float(threshold),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": best_params,
        "test_results": metrics,
    }

    Path("models/model_info.json").write_text(
        json.dumps(model_info, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nFINAL TEST RESULTS")
    print(json.dumps(metrics, indent=2))

    print("\nSAVED FILES")
    print("models/late_orders.joblib")
    print("models/model_info.json")
    print("reports/metrics.json")


if __name__ == "__main__":
    main()
