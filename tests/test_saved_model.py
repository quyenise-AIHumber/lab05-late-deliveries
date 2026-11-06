"""Verify a saved pipeline predicts identically."""

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import (
    build_pipeline,
    split_orders,
)


def test_saved_model_reloads_identically(tmp_path, orders):
    X_train, _, y_train, _ = split_orders(orders)

    pipeline = build_pipeline(LogisticRegression(max_iter=1000))
    pipeline.fit(X_train, y_train)

    sample = X_train.head(3).copy()

    # Test missing input and unknown category.
    sample.loc[sample.index[0], "distance_km"] = np.nan
    sample.loc[sample.index[1], "weather"] = "hail"
    sample["prep_minutes"] = np.nan

    before = pipeline.predict_proba(sample)

    model_path = tmp_path / "saved_model.joblib"
    joblib.dump(pipeline, model_path)

    restored = joblib.load(model_path)
    after = restored.predict_proba(sample)

    assert np.allclose(before, after)
    assert after.shape == (3, 2)
