import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from src import config
from src import viz
from src.simulation import (FixedInterval, NoRetrain, fit_model, prepare_data, run_simulation,
                            window_metrics)

plt = viz.plt
C = viz.COLORS


def decay_statistics(res: pd.DataFrame) -> dict:
    r = res.dropna(subset=["pr_auc"]).sort_values("window")
    x = r["window"].values - r["window"].min()
    out = {}
    for m in ("roc_auc", "pr_auc", "recall_at_1pct_fpr", "f1"):
        slope = stats.linregress(x, r[m].values)
        rho, p = stats.spearmanr(x, r[m].values)
        head, tail = r[m].values[:3].mean(), r[m].values[-3:].mean()
        out[m] = dict(first3_mean=float(head), last3_mean=float(tail),
                      relative_drop_pct=float(100 * (head - tail) / head) if head else None,
                      slope_per_week=float(slope.slope), spearman_rho=float(rho), spearman_p=float(p))
    worst = r.loc[r["pr_auc"].idxmin()]
    best = r.loc[r["pr_auc"].idxmax()]
    out["best_window"] = dict(window=int(best["window"]), pr_auc=float(best["pr_auc"]))
    out["worst_window"] = dict(window=int(worst["window"]), pr_auc=float(worst["pr_auc"]))
    return out


def evaluation_protocols(df, cols, stream):
    rows = []
    tr, te = train_test_split(df, test_size=0.2, stratify=df[config.TARGET], random_state=config.SEED)
    m = fit_model(tr[cols], tr[config.TARGET])
    p = m.predict_proba(te[cols])[:, 1]
    rows.append(dict(protocol="Random 80/20 split (static)", roc_auc=roc_auc_score(te[config.TARGET], p),
                     pr_auc=average_precision_score(te[config.TARGET], p), n_test=len(te)))
    cut = df[config.TIME_COL].quantile(0.8)
    tr, te = df[df[config.TIME_COL] <= cut], df[df[config.TIME_COL] > cut]
    m = fit_model(tr[cols], tr[config.TARGET])
    p = m.predict_proba(te[cols])[:, 1]
    rows.append(dict(protocol="Temporal 80/20 split", roc_auc=roc_auc_score(te[config.TARGET], p),
                     pr_auc=average_precision_score(te[config.TARGET], p), n_test=len(te)))
    rows.append(dict(protocol="Deploy-once, no retraining (pooled)",
                     roc_auc=roc_auc_score(stream["y"], stream["proba"]),
                     pr_auc=average_precision_score(stream["y"], stream["proba"]), n_test=len(stream)))
    late = stream[stream["window"] >= stream["window"].quantile(0.75)]
    rows.append(dict(protocol="Deploy-once, last 25% of weeks", roc_auc=roc_auc_score(late["y"], late["proba"]),
                     pr_auc=average_precision_score(late["y"], late["proba"]), n_test=len(late)))
    return pd.DataFrame(rows)


def compare_models(df, cols, initial_windows):
    import xgboost as xgb
    train = df[df["window"] < initial_windows - 1]
    val = df[df["window"] == initial_windows - 1]
    deploy = df[df["window"] >= initial_windows]
    pos = train[config.TARGET].sum()
    spw = (len(train) - pos) / pos
    models = {
        "LightGBM": fit_model(train[cols], train[config.TARGET]),
        "XGBoost": xgb.XGBClassifier(tree_method="hist", enable_categorical=True, n_estimators=300,
                                     learning_rate=0.05, max_depth=8, subsample=0.8, colsample_bytree=0.5,
                                     scale_pos_weight=spw, n_jobs=-1, random_state=config.SEED,
                                     eval_metric="aucpr").fit(train[cols], train[config.TARGET]),
    }
    rows = []
    for name, model in models.items():
        per = []
        for _, g in deploy.groupby("window"):
            pr = model.predict_proba(g[cols])[:, 1]
            per.append(window_metrics(g[config.TARGET].values, pr, 0.5))
        per = pd.DataFrame(per)
        rows.append(dict(model=name, mean_roc_auc=per["roc_auc"].mean(), mean_pr_auc=per["pr_auc"].mean(),
                         first4_pr_auc=per["pr_auc"].head(4).mean(), last4_pr_auc=per["pr_auc"].tail(4).mean()))
    return pd.DataFrame(rows)


def plot_decay(res, info_row=None, name="decay_curve", title="No retraining: performance over deployment weeks"):
    from matplotlib.ticker import MaxNLocator
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.8))
    x = res["window"]

    def line(a, m, col, lab, trend=True):
        a.plot(x, res[m], marker="o", ms=3, color=col, label=lab)
        ok = res[m].notna()
        if trend and ok.sum() > 2:
            z = np.polyfit(x[ok], res[m][ok], 1)
            a.plot(x, np.polyval(z, x), ls="--", lw=1, color=col, alpha=0.6)

    line(ax[0], "roc_auc", C["blue"], "ROC-AUC")
    ax[0].axhline(0.5, color=C["gray"], ls=":", lw=1)
    ax[0].set_title("ROC-AUC (0.5 = chance)")
    line(ax[1], "pr_auc", C["red"], "PR-AUC")
    line(ax[1], "recall_at_1pct_fpr", C["green"], "Recall @ 1% FPR")
    ax[1].set_title("PR-AUC and recall @ 1% FPR"); ax[1].legend(loc="upper right")
    line(ax[2], "precision", C["orange"], "Precision", trend=False)
    line(ax[2], "recall", C["purple"], "Recall", trend=False)
    line(ax[2], "f1", C["gray"], "F1", trend=False)
    ax[2].set_title("Fixed-threshold operating point"); ax[2].legend(loc="upper right")
    for a in ax:
        a.set_xlabel("Deployment week index"); a.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle(title, fontweight="bold", y=1.03)
    fig.tight_layout()
    return viz.save(fig, name)


def main(policies=("no_retrain",), compare=True, protocols=True):
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(df)
    print("prepared:", info)
    summary = {"data": info, "policies": {}}

    streams = {}
    for spec in policies:
        pol = NoRetrain() if spec == "no_retrain" else FixedInterval(int(spec.split(":")[1]))
        res, pinfo, stream, model = run_simulation(df, cols, pol)
        res.to_csv(config.TABLE_DIR / f"window_metrics_{pol.name}.csv", index=False)
        stream.to_parquet(config.RESULTS_DIR / f"stream_{pol.name}.parquet", index=False)
        streams[pol.name] = stream
        summary["policies"][pol.name] = dict(run=pinfo, decay=decay_statistics(res),
                                             mean_pr_auc=float(res["pr_auc"].mean()),
                                             mean_roc_auc=float(res["roc_auc"].mean()))
        plot_decay(res, name=f"decay_curve_{pol.name}" if pol.name != "no_retrain" else "decay_curve",
                   title=f"{pol.name}: performance over deployment weeks")
        if pol.name == "no_retrain":
            base_model = model

    imp = pd.DataFrame({"feature": cols, "gain": base_model.booster_.feature_importance("gain")})
    imp["gain_pct"] = 100 * imp["gain"] / imp["gain"].sum()
    imp = imp.sort_values("gain", ascending=False)
    imp.head(30).to_csv(config.TABLE_DIR / "top_features_gain.csv", index=False)
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top["feature"], top["gain_pct"], color=C["blue"]); ax.set_xlabel("Share of total gain (%)")
    ax.set_title("Baseline LightGBM: top-20 features")
    viz.save(fig, "top_features")

    if protocols:
        prot = evaluation_protocols(df, cols, streams["no_retrain"])
        prot.to_csv(config.TABLE_DIR / "evaluation_protocols.csv", index=False)
        summary["evaluation_protocols"] = prot.round(4).to_dict("records")
        fig, ax = plt.subplots(figsize=(7.5, 3.6))
        yy = np.arange(len(prot))
        ax.barh(yy - 0.2, prot["roc_auc"], 0.4, color=C["blue"], label="ROC-AUC")
        ax.barh(yy + 0.2, prot["pr_auc"], 0.4, color=C["red"], label="PR-AUC")
        ax.set_yticks(yy); ax.set_yticklabels(prot["protocol"]); ax.invert_yaxis(); ax.legend()
        ax.set_title("Same model, different evaluation protocol")
        viz.save(fig, "evaluation_protocols")
    if compare:
        try:
            cmp_ = compare_models(df, cols, info["initial_train_windows"])
            cmp_.to_csv(config.TABLE_DIR / "model_comparison.csv", index=False)
            summary["model_comparison"] = cmp_.round(4).to_dict("records")
        except Exception as e:
            print("model comparison skipped:", e)

    (config.RESULTS_DIR / "baseline_summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps(summary["policies"]["no_retrain"]["decay"]["pr_auc"], indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="+", default=["no_retrain"], help="no_retrain fixed:4 ...")
    ap.add_argument("--skip-compare", action="store_true")
    ap.add_argument("--skip-protocols", action="store_true")
    a = ap.parse_args()
    main(a.policies, compare=not a.skip_compare, protocols=not a.skip_protocols)
