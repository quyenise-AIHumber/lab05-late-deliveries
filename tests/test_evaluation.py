import numpy as np

from latedeliveries.evaluation import (
    choose_threshold,
    order_cost,
)


def test_cost_hand_calculated():
    # Two vouchers ($8) and one missed late ($15).
    assert (
        order_cost(
            [1, 0, 1, 0],
            [True, True, False, False],
        )
        == 23
    )


def test_threshold_can_flag_nobody():
    labels = [0, 0, 0, 1]
    probabilities = np.array([0.40, 0.30, 0.20, 0.01])
    threshold, costs = choose_threshold(labels, probabilities)

    assert (probabilities >= threshold).sum() == 0
    assert costs["cost"].min() == 15


def test_threshold_can_flag_everybody():
    labels = [1, 1, 1, 0]
    probabilities = np.array([0.40, 0.50, 0.60, 0.70])
    threshold, costs = choose_threshold(labels, probabilities)

    assert (probabilities >= threshold).sum() == 4
    assert costs["cost"].min() == 16
