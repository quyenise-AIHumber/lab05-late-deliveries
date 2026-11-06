"""Model comparison and training-only selection."""

from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier

from latedeliveries.data import make_orders
from latedeliveries.evaluation import choose_threshold, order_cost
from latedeliveries.pipeline import (
    RANDOM_STATE,
    build_pipeline,
    split_orders,
)


def compare_and_tune():
    """Compare and tune models without evaluating the test set."""
    Path("reports").mkdir(exist_ok=True)

    orders = make_orders()
    X_train, _, y_train, _ = split_orders(orders)

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    models = {
        "Dummy": DummyClassifier(strategy="prior"),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "DecisionTreeBalanced": DecisionTreeClassifier(
            max_depth=5,
            min_samples_leaf=10,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=80,
            min_samples_leaf=8,
            random_state=RANDOM_STATE,
        ),
    }

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    cv_rows = []
    cost_rows = []

    # Compare the baseline and three candidate models.
    for name, model in models.items():
        pipeline = build_pipeline(model)

        scores = cross_validate(
            pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=1,
        )

        row = {"model": name}

        for metric in scoring:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = round(float(values.mean()), 4)
            row[f"{metric}_std"] = round(float(values.std()), 4)

        cv_rows.append(row)

        # Evaluate business cost using out-of-fold
        # training predictions, not the test set.
        if name != "Dummy":
            probabilities = cross_val_predict(
                pipeline,
                X_train,
                y_train,
                cv=cv,
                method="predict_proba",
                n_jobs=1,
            )[:, 1]

            threshold, cost_table = choose_threshold(y_train, probabilities)

            cost_rows.append(
                {
                    "model": name,
                    "cv_cost": int(cost_table["cost"].min()),
                    "threshold": threshold,
                }
            )

    comparison = pd.DataFrame(cv_rows)
    comparison.to_csv("reports/cv_results.csv", index=False)

    print("\nFIVE-FOLD CROSS-VALIDATION RESULTS")
    print(comparison.to_string(index=False))

    cost_comparison = pd.DataFrame(cost_rows)
    cost_comparison = cost_comparison.sort_values(["cv_cost", "model"])
    cost_comparison.to_csv(
        "reports/model_cost_comparison.csv",
        index=False,
    )

    print("\nOUT-OF-FOLD BUSINESS COST COMPARISON")
    print(cost_comparison.to_string(index=False))

    # Choose using business cost, not accuracy alone.
    winner_name = str(cost_comparison.iloc[0]["model"])

    # At least two hyperparameters and six combinations.
    grids = {
        "LogisticRegression": {
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        "DecisionTreeBalanced": {
            "model__max_depth": [3, 5, 8],
            "model__min_samples_leaf": [5, 10],
        },
        "RandomForest": {
            "model__n_estimators": [40, 80, 120],
            "model__min_samples_leaf": [4, 8],
        },
    }

    search = GridSearchCV(
        build_pipeline(models[winner_name]),
        param_grid=grids[winner_name],
        scoring="roc_auc",
        cv=cv,
        n_jobs=1,
        refit=True,
    )
    search.fit(X_train, y_train)

    old = comparison.loc[comparison["model"] == winner_name].iloc[0]

    gain = search.best_score_ - old["roc_auc_mean"]
    spread = old["roc_auc_std"]

    print(f"\nCHOSEN MODEL: {winner_name}")
    print(f"Best settings: {search.best_params_}")
    print(f"Untuned mean ROC-AUC: {old['roc_auc_mean']:.4f}")
    print(f"Untuned fold-to-fold SD: {spread:.4f}")
    print(f"Tuned mean ROC-AUC: {search.best_score_:.4f}")
    print(f"Improvement exceeds fold spread: {gain > spread}")

    # Tune the threshold using training-only
    # out-of-fold predicted probabilities.
    probabilities = cross_val_predict(
        search.best_estimator_,
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    threshold, costs = choose_threshold(y_train, probabilities)
    costs.to_csv("reports/threshold_costs.csv", index=False)

    none_cost = order_cost(y_train, [False] * len(y_train))
    all_cost = order_cost(y_train, [True] * len(y_train))

    print(f"\nBEST THRESHOLD: {threshold:.2f}")
    print(f"Model CV cost: ${int(costs['cost'].min())}")
    print(f"Flag nobody cost: ${none_cost}")
    print(f"Flag everybody cost: ${all_cost}")
    print("\nTest set has NOT been evaluated.")

    return search.best_estimator_, threshold


if __name__ == "__main__":
    compare_and_tune()
