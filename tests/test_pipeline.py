import numpy as np
import pandas as pd

from src import config, cost
from src.simulation import (DriftTriggered, FixedInterval, NoRetrain, run_simulation, training_slice)


def test_cleaning(clean_df):
    assert config.TARGET in clean_df and "has_identity" in clean_df
    assert "dead_col" not in clean_df.columns
    assert clean_df[config.ID_COL].is_unique
    assert clean_df[config.TIME_COL].is_monotonic_increasing


def test_features_exclude_time_and_target(prepared):
    _, cols, _ = prepared
    assert config.TIME_COL not in cols and config.TARGET not in cols and config.ID_COL not in cols
    assert "window" not in cols


def test_training_slice_has_no_future_rows(prepared):
    df, _, _ = prepared
    t0 = df[config.TIME_COL].min()
    for w in (9, 15):
        for delay in (0, 7, 14):
            tr = training_slice(df, w, delay)
            w_start = t0 + w * config.WINDOW_DAYS * config.SECONDS_PER_DAY
            assert tr["window"].max() < w
            assert tr[config.TIME_COL].max() <= w_start - delay * config.SECONDS_PER_DAY


def test_fixed_interval_schedule():
    p = FixedInterval(4)
    assert [w for w in range(8, 26) if p.should_retrain(w, 8)] == [12, 16, 20, 24]
    assert not NoRetrain().should_retrain(20, 8)


def test_drift_policy_triggers_on_error_increase_only():
    rng = np.random.default_rng(0)
    y = np.ones(3000, dtype=int)
    pol = DriftTriggered("adwin_miss", cooldown=0)
    for w in range(3):
        pred = (rng.random(3000) > 0.10).astype(int)
        pol.observe(w, y, pred, pred.astype(float))
    assert not pol.should_retrain(3, 0)
    pred = (rng.random(3000) > 0.80).astype(int)
    pol.observe(3, y, pred, pred.astype(float))
    assert pol.should_retrain(4, 0) is True
    assert pol.should_retrain(5, 0) is False
    pol2 = DriftTriggered("adwin_miss", cooldown=0)
    pol2.observe(0, y, (rng.random(3000) > 0.6).astype(int), None)
    pol2.observe(1, y, (rng.random(3000) > 0.02).astype(int), None)
    assert not pol2.should_retrain(2, 0)


def test_cost_model():
    s = pd.DataFrame({"y": [1, 1, 0, 0], "pred": [1, 0, 1, 0], "amount": [100.0, 50.0, 10.0, 10.0]})
    c = cost.components(s, n_retrains=2)
    assert c["missed_amount"] == 50 and c["flagged"] == 2 and c["false_alarms"] == 1
    t = cost.total_cost(c, review_cost=1, retrain_cost=10, fraud_loss_rate=1)
    assert t["total"] == 50 + 2 + 20
    base = cost.components(pd.DataFrame({"y": [1, 1, 0, 0], "pred": [0, 0, 0, 0], "amount": [100.0, 50.0, 10.0, 10.0]}), 0)
    be = cost.break_even_retrain_cost(c, base, review_cost=1, fraud_loss_rate=1)
    assert be == ((150 + 0) - (50 + 2)) / 2


def test_end_to_end_small(prepared):
    df, cols, info = prepared
    res, rinfo, stream, _ = run_simulation(df, cols, NoRetrain(), verbose=False)
    assert len(res) == info["n_windows"] - info["initial_train_windows"]
    assert {"roc_auc", "pr_auc", "recall_at_1pct_fpr", "f1"} <= set(res.columns)
    assert rinfo["n_retrains"] == 0
    assert stream["window"].min() == info["initial_train_windows"]
    res2, rinfo2, _, _ = run_simulation(df, cols, FixedInterval(4), verbose=False)
    assert rinfo2["retrain_windows"] == [12, 16, 20, 24]
    a, b = res.set_index("window"), res2.set_index("window")
    assert np.allclose(a.loc[8:11, "roc_auc"], b.loc[8:11, "roc_auc"], equal_nan=True)
