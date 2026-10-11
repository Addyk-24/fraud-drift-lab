import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config
from src.features import add_time_features, candidate_features, select_features


def prepare_data(df: pd.DataFrame, window_days=config.WINDOW_DAYS,
                 initial_windows=config.INITIAL_TRAIN_WINDOWS):
    df = add_time_features(df.copy())
    t0 = df[config.TIME_COL].min()
    df["window"] = ((df[config.TIME_COL] - t0) // (window_days * config.SECONDS_PER_DAY)).astype("int16")
    cols = candidate_features(df.drop(columns=["window"]))
    init = df[df["window"] < initial_windows]
    kept, dropped = select_features(init, cols)
    info = {"n_candidate_features": len(cols), "n_selected_features": len(kept),
            "n_dropped_correlated": len(dropped), "n_windows": int(df["window"].max()) + 1,
            "window_days": window_days, "initial_train_windows": initial_windows}
    return df, kept, info


def fit_model(X, y, params=None):
    params = dict(params or config.LGBM_PARAMS)
    pos = max(int(y.sum()), 1)
    params["scale_pos_weight"] = (len(y) - pos) / pos
    return lgb.LGBMClassifier(**params).fit(X, y)


def best_f1_threshold(y, proba):
    grid = np.unique(np.quantile(proba, np.linspace(0.80, 0.9995, 200)))
    scores = [f1_score(y, proba >= g, zero_division=0) for g in grid]
    return float(grid[int(np.argmax(scores))])


def window_metrics(y, proba, thr, amount=None):
    out = dict(n=len(y), n_fraud=int(y.sum()), fraud_rate=float(np.mean(y)))
    if 0 < y.sum() < len(y):
        out["roc_auc"] = roc_auc_score(y, proba)
        out["pr_auc"] = average_precision_score(y, proba)
        fpr, tpr, _ = roc_curve(y, proba)
        out["recall_at_1pct_fpr"] = float(np.interp(0.01, fpr, tpr))
    else:
        out.update(roc_auc=np.nan, pr_auc=np.nan, recall_at_1pct_fpr=np.nan)
    pred = proba >= thr
    out["precision"] = precision_score(y, pred, zero_division=0)
    out["recall"] = recall_score(y, pred, zero_division=0)
    out["f1"] = f1_score(y, pred, zero_division=0)
    if amount is not None and (y * amount).sum() > 0:
        out["amount_recall"] = float(((pred & (y == 1)) * amount).sum() / (y * amount).sum())
    else:
        out["amount_recall"] = np.nan
    return out


class Policy:
    name = "policy"

    def should_retrain(self, w: int, first_deploy: int) -> bool:
        raise NotImplementedError

    def observe(self, w: int, y, pred, proba):
        pass


    def notify_retrained(self, w: int):
        pass


class NoRetrain(Policy):
    name = "no_retrain"

    def should_retrain(self, w, first_deploy):
        return False


@dataclass
class FixedInterval(Policy):
    k: int = 4
    name: str = field(init=False)

    def __post_init__(self):
        self.name = f"fixed_{self.k}w"

    def should_retrain(self, w, first_deploy):
        return w > first_deploy and (w - first_deploy) % self.k == 0


class DriftTriggered(Policy):

    def __init__(self, detector="adwin_miss", cooldown=2, delta=0.002, perf_drop=0.15, ref_windows=3):
        self.detector, self.cooldown, self.delta = detector, cooldown, delta
        self.perf_drop, self.ref_windows = perf_drop, ref_windows
        self.name = f"drift_{detector}"
        self.triggers, self._flag = [], False
        self._reset()

    def _reset(self):
        from river import drift
        self._age, self._ref, self._n, self._sum = 0, [], 0, 0.0
        if self.detector.startswith("adwin"):
            self._det = drift.ADWIN(delta=self.delta)
        elif self.detector.startswith("ddm"):
            try:
                self._det = drift.binary.DDM()
            except AttributeError:
                self._det = drift.DDM()
        else:
            self._det = None

    def should_retrain(self, w, first_deploy):
        flag, self._flag = self._flag, False
        return flag

    def notify_retrained(self, w):
        self._reset()

    def observe(self, w, y, pred, proba):
        self._age += 1
        if self.detector == "perf_drop":
            from sklearn.metrics import average_precision_score
            if y.sum() == 0:
                return
            ap = average_precision_score(y, proba)
            if len(self._ref) < self.ref_windows:
                self._ref.append(ap)
            elif self._age > self.cooldown and ap < (1 - self.perf_drop) * np.mean(self._ref):
                self._flag = True
                self.triggers.append(w)
            return
        err = (np.asarray(pred) != np.asarray(y)).astype(int)
        if self.detector.endswith("_miss"):
            err = err[np.asarray(y) == 1]
        is_adwin = self.detector.startswith("adwin")
        for e in err:
            self._det.update(float(e) if is_adwin else bool(e))
            self._n += 1
            self._sum += e
            if self._det.drift_detected:
                if is_adwin and self._det.estimation <= self._sum / max(self._n, 1):
                    continue
                if self._age >= self.cooldown:
                    self._flag = True
                    self.triggers.append(w)
                    return


def training_slice(df, w, delay_days, mode="expanding", sliding_windows=8):
    t0 = df[config.TIME_COL].min()
    w_start = t0 + w * config.WINDOW_DAYS * config.SECONDS_PER_DAY
    known = df[config.TIME_COL] <= (w_start - delay_days * config.SECONDS_PER_DAY)
    m = known & (df["window"] < w)
    if mode == "sliding":
        m &= df["window"] >= w - sliding_windows
    return df[m]


def train_with_threshold(train, cols):
    last = train["window"].max()
    fit_df, val_df = train[train["window"] < last], train[train["window"] == last]
    model = fit_model(fit_df[cols], fit_df[config.TARGET])
    thr = best_f1_threshold(val_df[config.TARGET].values, model.predict_proba(val_df[cols])[:, 1])
    return model, thr


def run_simulation(df, cols, policy: Policy, initial_windows=config.INITIAL_TRAIN_WINDOWS,
                   delay_days=config.LABEL_DELAY_DAYS, mode="expanding", verbose=True):
    n_win = int(df["window"].max()) + 1
    first_deploy = initial_windows
    t = time.time()
    model, thr = train_with_threshold(training_slice(df, first_deploy, delay_days, mode), cols)
    train_secs = [time.time() - t]
    retrain_at = [first_deploy]
    rows, stream = [], []

    for w in range(first_deploy, n_win):
        if policy.should_retrain(w, first_deploy):
            t = time.time()
            model, thr = train_with_threshold(training_slice(df, w, delay_days, mode), cols)
            train_secs.append(time.time() - t)
            retrain_at.append(w)
            policy.notify_retrained(w)
        cur =df[df["window"] == w]
        if len(cur) == 0:
            continue
        y = cur[config.TARGET].values
        proba = model.predict_proba(cur[cols])[:, 1]
        amt = cur["TransactionAmt"].values if "TransactionAmt" in cur else None
        m = window_metrics(y, proba, thr, amt)
        m.update(window=w, policy=policy.name, retrained_before=int(w in retrain_at and w != first_deploy),
                 threshold=thr)
        rows.append(m)
        policy.observe(w, y, proba >= thr, proba)
        stream.append(pd.DataFrame({"window": w, "TransactionDT": cur[config.TIME_COL].values, "y": y,
                                    "proba": proba, "pred": (proba >= thr).astype("int8"),
                                    "amount": amt if amt is not None else np.nan}))
        if verbose:
            print(f"[{policy.name}] window {w:>2}  n={m['n']:>6}  fraud={m['n_fraud']:>4}  "
                  f"AUC={m['roc_auc']:.3f}  PR-AUC={m['pr_auc']:.3f}  F1={m['f1']:.3f}")

    res = pd.DataFrame(rows)
    info = dict(policy=policy.name, n_retrains=len(retrain_at) - 1, retrain_windows=retrain_at[1:],
                total_train_seconds=float(sum(train_secs)), label_delay_days=delay_days, train_mode=mode,
                trigger_windows=[int(x) for x in getattr(policy, "triggers", [])])
    return res, info, pd.concat(stream, ignore_index=True), model
