import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from river import drift
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from src import config, viz
from src.simulation import prepare_data

plt = viz.plt
C = viz.COLORS


def _make_ddm():
    try:
        return drift.binary.DDM()
    except AttributeError:
        return drift.DDM()


def run_detectors(stream: pd.DataFrame) -> pd.DataFrame:
    t0 = stream["TransactionDT"].values[0]
    rows = []
    for suffix, sub in (("", stream), ("-miss", stream[stream["y"] == 1])):
        err = (sub["pred"].values != sub["y"].values).astype(int)
        win, tt = sub["window"].values, sub["TransactionDT"].values
        for base, det, cast in (("ADWIN", drift.ADWIN(), float), ("DDM", _make_ddm(), bool)):
            for i, e in enumerate(err):
                det.update(cast(e))
                if det.drift_detected:
                    rows.append(dict(detector=base + suffix, stream_index=i, window=int(win[i]),
                                     day=float((tt[i] - t0) / config.SECONDS_PER_DAY)))
    return pd.DataFrame(rows, columns=["detector", "stream_index", "window", "day"])


def degradation_onset(win_metrics: pd.DataFrame, drop=0.20, patience=2):
    r = win_metrics.sort_values("window").reset_index(drop=True)
    ref = r["pr_auc"].head(3).mean()
    bad = (r["pr_auc"] < (1 - drop) * ref).astype(int).values
    for i in range(len(bad) - patience + 1):
        if bad[i:i + patience].all():
            return int(r.loc[i, "window"])
    return None


def psi(ref: pd.Series, cur: pd.Series, bins=10, eps=1e-4) -> float:
    if pd.api.types.is_numeric_dtype(ref):
        edges = np.unique(np.nanquantile(ref, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            return 0.0
        edges[0], edges[-1] = -np.inf, np.inf
        rc = pd.cut(ref, edges).value_counts(normalize=True, dropna=False)
        cc = pd.cut(cur, edges).value_counts(normalize=True, dropna=False)
    else:
        rc = ref.astype(object).fillna("NA").value_counts(normalize=True)
        cc = cur.astype(object).fillna("NA").value_counts(normalize=True)
    idx = rc.index.union(cc.index)
    r = rc.reindex(idx, fill_value=0).clip(lower=eps).values
    c = cc.reindex(idx, fill_value=0).clip(lower=eps).values
    return float(np.sum((c - r) * np.log(c / r)))


def ks_stat(ref, cur):
    if not pd.api.types.is_numeric_dtype(ref):
        return np.nan
    a, b = ref.dropna().values, cur.dropna().values
    return float(stats.ks_2samp(a, b).statistic) if len(a) and len(b) else np.nan


def adversarial_auc(ref: pd.DataFrame, cur: pd.DataFrame, cols, n=8000):
    a = ref.sample(min(n, len(ref)), random_state=config.SEED)
    b = cur.sample(min(n, len(cur)), random_state=config.SEED)
    X = pd.concat([a[cols], b[cols]])
    y = np.r_[np.zeros(len(a)), np.ones(len(b))]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, stratify=y, random_state=config.SEED)
    m = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.1, num_leaves=31, colsample_bytree=0.5,
                           verbose=-1, n_jobs=-1, random_state=config.SEED).fit(Xtr, ytr)
    return float(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))


def main(top_n=20):
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(df)
    init_w = info["initial_train_windows"]
    ref = df[df["window"] < init_w]

    stream = pd.read_parquet(config.RESULTS_DIR / "stream_no_retrain.parquet")
    wm = pd.read_csv(config.TABLE_DIR / "window_metrics_no_retrain.csv")
    top = pd.read_csv(config.TABLE_DIR / "top_features_gain.csv")["feature"].head(top_n).tolist()

    det = run_detectors(stream)
    det.to_csv(config.TABLE_DIR / "drift_detections.csv", index=False)
    onset = degradation_onset(wm)
    summary = {"degradation_onset_window": onset, "detectors": {}}
    for name in ("ADWIN", "DDM", "ADWIN-miss", "DDM-miss"):
        d = det[det["detector"] == name]
        first = int(d["window"].min()) if len(d) else None
        summary["detectors"][name] = dict(
            n_detections=int(len(d)), first_detection_window=first,
            detection_windows=sorted(set(d["window"].astype(int))),
            delay_vs_onset_weeks=(first - onset) if (first is not None and onset is not None) else None)

    deploy_windows = sorted(df.loc[df["window"] >= init_w, "window"].unique())
    psi_rows, ks_rows, adv_rows = {}, {}, []
    for w in deploy_windows:
        cur = df[df["window"] == w]
        psi_rows[w] = {f: psi(ref[f], cur[f]) for f in top}
        ks_rows[w] = {f: ks_stat(ref[f], cur[f]) for f in top}
        adv_rows.append(dict(window=int(w), adversarial_auc=adversarial_auc(ref, cur, cols)))
    psi_df = pd.DataFrame(psi_rows).T
    ks_df = pd.DataFrame(ks_rows).T
    adv = pd.DataFrame(adv_rows)
    psi_df.to_csv(config.TABLE_DIR / "psi_top_features.csv")
    ks_df.to_csv(config.TABLE_DIR / "ks_top_features.csv")
    adv.to_csv(config.TABLE_DIR / "adversarial_drift.csv", index=False)

    summary["psi"] = dict(
        n_features_psi_over_0_25_in_last_window=int((psi_df.iloc[-1] > 0.25).sum()),
        n_features_psi_over_0_10_in_last_window=int((psi_df.iloc[-1] > 0.10).sum()),
        most_shifted_features=psi_df.iloc[-4:].mean().sort_values(ascending=False).head(5).round(4).to_dict())
    summary["adversarial_auc"] = dict(first_deployment_week=float(adv["adversarial_auc"].iloc[0]),
                                      last_deployment_week=float(adv["adversarial_auc"].iloc[-1]),
                                      max=float(adv["adversarial_auc"].max()))
    summary["corr_adversarial_auc_vs_pr_auc"] = float(
        np.corrcoef(adv["adversarial_auc"], wm.sort_values("window")["pr_auc"].values[:len(adv)])[0, 1])

    fig, ax = plt.subplots(figsize=(9, 3.8))
    wms = wm.sort_values("window")
    ax.plot(wms["window"], wms["pr_auc"], marker="o", ms=3, color=C["red"], label="PR-AUC (no retraining)")
    styles = {"ADWIN": (C["blue"], "--"), "DDM": (C["green"], ":"),
              "ADWIN-miss": (C["orange"], "-."), "DDM-miss": (C["purple"], (0, (1, 3)))}
    for name, (col, ls) in styles.items():
        for i, w in enumerate(summary["detectors"][name]["detection_windows"]):
            ax.axvline(w, color=col, ls=ls, alpha=0.6, label=f"{name} detection" if i == 0 else None)
    if onset is not None:
        ax.axvline(onset, color="black", lw=1.5, label="Degradation onset (ref.)")
    ax.set_xlabel("Deployment week index"); ax.set_ylabel("PR-AUC")
    ax.set_title("Drift detectors on the error stream vs measured degradation"); ax.legend()
    viz.save(fig, "drift_detection")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(np.clip(psi_df.T.values, 0, 1), aspect="auto", cmap="magma_r", vmin=0, vmax=1)
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    ax.set_xticks(range(len(deploy_windows))[::2]); ax.set_xticklabels(deploy_windows[::2])
    ax.set_xlabel("Deployment week index"); ax.grid(False)
    ax.set_title("PSI of top features vs reference period (>0.25 = major shift)")
    fig.colorbar(im, ax=ax, label="PSI (clipped at 1)")
    viz.save(fig, "psi_heatmap")

    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.plot(adv["window"], adv["adversarial_auc"], marker="o", ms=3, color=C["purple"])
    ax.axhline(0.5, color=C["gray"], ls="--", lw=1)
    ax.set_xlabel("Deployment week index"); ax.set_ylabel("Adversarial AUC")
    ax.set_title("How separable is week w from the training period? (0.5 = identical)")
    viz.save(fig, "adversarial_drift")

    (config.RESULTS_DIR / "drift_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
