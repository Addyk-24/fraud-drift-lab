import time
from dataclasses import dataclass, field

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)

from src import config
from src.driftstats import psi
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
    df.attrs["window_days"] = window_days
    return df, kept, info


def fit_model(X, y, params=None, class_weight=True):
    params = dict(params or config.LGBM_PARAMS)
    if class_weight:
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

    def observe_data(self, w: int, cur):
        pass

    def on_train(self, train, model, cols):
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


class HybridDrift(DriftTriggered):

    def __init__(self, detector="adwin", max_age=8, **kw):
        super().__init__(detector=detector, **kw)
        self.max_age = max_age
        self.name = f"hybrid_{detector}_max{max_age}"

    def should_retrain(self, w, first_deploy):
        flag, self._flag = self._flag, False
        if flag:
            return True
        if self._age >= self.max_age:
            self.triggers.append(w)
            return True
        return False


class CovariateDrift(Policy):

    def __init__(self, min_features=3, psi_threshold=0.25, top_n=20, ref_windows=4, cooldown=2):
        self.min_features, self.psi_threshold = min_features, psi_threshold
        self.top_n, self.ref_windows, self.cooldown = top_n, ref_windows, cooldown
        self.name = f"drift_psi{min_features}"
        self.triggers, self._flag, self._age = [], False, 0
        self._ref, self._feats = None, []

    def on_train(self, train, model, cols):
        gain = pd.Series(model.booster_.feature_importance("gain"), index=cols)
        self._feats = list(gain.nlargest(self.top_n).index)
        last = train["window"].max()
        self._ref = train[train["window"] > last - self.ref_windows][self._feats]
        self._age = 0

    def should_retrain(self, w, first_deploy):
        flag, self._flag = self._flag, False
        return flag

    def observe_data(self, w, cur):
        self._age += 1
        if self._ref is None:
            return
        n_shifted = sum(psi(self._ref[f], cur[f]) > self.psi_threshold for f in self._feats)
        if n_shifted >= self.min_features and self._age >= self.cooldown:
            self._flag = True
            self.triggers.append(w)


def training_slice(df, w, delay_days, mode="expanding", sliding_windows=8):
    t0 = df[config.TIME_COL].min()
    window_days = df.attrs.get("window_days", config.WINDOW_DAYS)
    w_start = t0 + w * window_days * config.SECONDS_PER_DAY
    known = df[config.TIME_COL] <= (w_start - delay_days * config.SECONDS_PER_DAY)
    m = known & (df["window"] < w)
    if mode == "sliding":
        m &= df["window"] >= w - sliding_windows
    return df[m]


def train_with_threshold(train, cols, params=None, class_weight=True):
    last = train["window"].max()
    fit_df, val_df = train[train["window"] < last], train[train["window"] == last]
    model = fit_model(fit_df[cols], fit_df[config.TARGET], params, class_weight)
    thr = best_f1_threshold(val_df[config.TARGET].values, model.predict_proba(val_df[cols])[:, 1])
    return model, thr


def run_simulation(df, cols, policy: Policy, initial_windows=config.INITIAL_TRAIN_WINDOWS,
                   delay_days=config.LABEL_DELAY_DAYS, mode="expanding", verbose=True,
                   params=None, class_weight=True):
    n_win = int(df["window"].max()) + 1
    first_deploy = initial_windows
    t = time.time()
    train0 = training_slice(df, first_deploy, delay_days, mode)
    model, thr = train_with_threshold(train0, cols, params, class_weight)
    policy.on_train(train0, model, cols)
    train_secs = [time.time() - t]
    retrain_at = [first_deploy]
    rows, stream = [], []

    for w in range(first_deploy, n_win):
        if policy.should_retrain(w, first_deploy):
            t = time.time()
            trainw = training_slice(df, w, delay_days, mode)
            model, thr = train_with_threshold(trainw, cols, params, class_weight)
            train_secs.append(time.time() - t)
            retrain_at.append(w)
            policy.notify_retrained(w)
            policy.on_train(trainw, model, cols)
        cur = df[df["window"] == w]
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
        policy.observe_data(w, cur)
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
