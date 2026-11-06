import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)


@pytest.mark.parametrize(
    "excluded",
    [
        "order_id",
        "prep_minutes",
        "customer_rating",
        "late",
        "vehicle",
        "courier_months",
    ],
)
def test_excluded_features(excluded):
    assert excluded not in FEATURES


def test_split_is_stratified_and_repeatable(orders):
    X1, X_test1, y1, y_test1 = split_orders(orders)
    X2, X_test2, y2, y_test2 = split_orders(orders)

    assert RANDOM_STATE == 42
    assert len(X1) == 2250
    assert len(X_test1) == 750
    assert X1.equals(X2)
    assert X_test1.equals(X_test2)
    assert y1.equals(y2)
    assert y_test1.equals(y_test2)
    assert abs(y1.mean() - y_test1.mean()) < 0.02


def test_probabilities_are_valid(orders):
    X_train, _, y_train, _ = split_orders(orders)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_train.head(5))

    assert probabilities.shape == (5, 2)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_missing_and_unknown_inputs(orders):
    X_train, _, y_train, _ = split_orders(orders)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    sample = X_train.head(1).copy()
    sample.loc[sample.index[0], "distance_km"] = np.nan
    sample.loc[sample.index[0], "weather"] = "hail"
    sample["prep_minutes"] = np.nan

    result = pipeline.predict_proba(sample)

    assert result.shape == (1, 2)
    assert np.isfinite(result).all()
