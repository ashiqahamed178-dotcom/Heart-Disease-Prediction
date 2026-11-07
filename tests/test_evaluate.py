from __future__ import annotations

import numpy as np
import pytest

from heartrisk.evaluate import (
    bootstrap_ci,
    choose_threshold,
    expected_calibration_error,
    threshold_metrics,
)


def test_choose_threshold_meets_target_recall() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 500)
    p = np.clip(0.3 * y + rng.random(500) * 0.7, 0, 1)
    t = choose_threshold(y, p, 0.9)
    assert threshold_metrics(y, p, t)["recall"] >= 0.9
    # A slightly higher threshold should fall below the target.
    assert threshold_metrics(y, p, t + 1e-9)["recall"] < 0.9


def test_choose_threshold_needs_positives() -> None:
    with pytest.raises(ValueError):
        choose_threshold(np.zeros(5), np.linspace(0, 1, 5), 0.9)


def test_threshold_metrics_counts() -> None:
    m = threshold_metrics([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
    assert (m["tp"], m["fp"], m["tn"], m["fn"]) == (1, 1, 1, 1)
    assert m["recall"] == 0.5
    assert m["specificity"] == 0.5


def test_ece_zero_for_perfect_calibration() -> None:
    y = np.array([0] * 50 + [1] * 50)
    p = np.full(100, 0.5)
    assert expected_calibration_error(y, p) == pytest.approx(0.0)


def test_bootstrap_ci_is_sensible() -> None:
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(0.4 * y + rng.random(300) * 0.6, 0, 1)
    lo, hi = bootstrap_ci(y, p, 0.5, n_boot=200)["roc_auc"]
    assert 0.5 < lo < hi <= 1.0
