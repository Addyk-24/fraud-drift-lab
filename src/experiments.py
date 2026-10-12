import argparse
import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import ParameterSampler

from src import config, cost, viz
from src.policy_comparison import make_policy
from src.simulation import fit_model, prepare_data, run_simulation

plt = viz.plt
C = viz.COLORS

EXTRA_DIR = config.RESULTS_DIR / "extra"
SPACE = dict(
    num_leaves=[31, 64, 128, 255],
    learning_rate=[0.03, 0.05, 0.1],
    n_estimators=[200, 300, 400],
    colsample_bytree=[0.3, 0.5, 0.7],
    min_child_samples=[20, 50, 100],
    reg_lambda=[0.0, 1.0, 5.0, 10.0],
    subsample=[0.7, 0.8, 1.0],
)


def tune(df, cols, info, n_trials=8):
    iw = info["initial_train_windows"]
    train = df[df["window"] < iw - 2]
    val = df[(df["window"] >= iw - 2) & (df["window"] < iw)]
    trials = [dict(config.LGBM_PARAMS)]
    for p in ParameterSampler(SPACE, n_trials, random_state=config.SEED):
        trials.append({**config.LGBM_PARAMS, **p})
    rows = []
    for i, params in enumerate(trials):
        t = time.time()
        m = fit_model(train[cols], train[config.TARGET], params)
        pr = average_precision_score(val[config.TARGET], m.predict_proba(val[cols])[:, 1])
        rows.append({"trial": i, "is_default": i == 0, "val_pr_auc": pr, "fit_seconds": time.time() - t,
                     **{k: params[k] for k in SPACE}})
        print(f"[tune {i}] val PR-AUC={pr:.4f} {time.time() - t:.0f}s", flush=True)
    tab = pd.DataFrame(rows).sort_values("val_pr_auc", ascending=False)
    tab.to_csv(config.TABLE_DIR / "tuning_trials.csv", index=False)
    best = trials[int(tab.iloc[0]["trial"])]
    (config.RESULTS_DIR / "best_params.json").write_text(json.dumps({k: best[k] for k in best}, indent=1, default=float))
    return best, tab


def summarise(res, stream, pinfo):
    comp = cost.components(stream, pinfo["n_retrains"])
    tc = cost.total_cost(comp)
    return dict(mean_pr_auc=res["pr_auc"].mean(), mean_roc_auc=res["roc_auc"].mean(),
                mean_recall_at_1pct_fpr=res["recall_at_1pct_fpr"].mean(), n_retrains=pinfo["n_retrains"],
                train_seconds=pinfo["total_train_seconds"], total_cost=tc["total"], n_weeks=len(res))


def run_one(group, variant, label, spec, df, cols, info, params=None, class_weight=True, force=False):
    EXTRA_DIR.mkdir(parents=True, exist_ok=True)
    f = EXTRA_DIR / f"{group}__{variant}__{label}.json"
    if f.exists() and not force:
        return json.loads(f.read_text())
    pol, delay, mode = make_policy(spec)
    res, pinfo, stream, _ = run_simulation(df, cols, pol, initial_windows=info["initial_train_windows"],
                                           delay_days=delay, mode=mode, verbose=False, params=params,
                                           class_weight=class_weight)
    row = dict(group=group, variant=variant, policy=label, spec=spec, **summarise(res, stream, pinfo))
    f.write_text(json.dumps(row, indent=1, default=float))
    print(f"[done] {group}/{variant}/{label}: PR-AUC={row['mean_pr_auc']:.4f} retrains={row['n_retrains']} "
          f"{row['train_seconds']:.0f}s", flush=True)
    return row


def default_row(group, variant, label, policy_name):
    res = pd.read_csv(config.TABLE_DIR / f"window_metrics_{policy_name}.csv")
    stream = pd.read_parquet(config.RESULTS_DIR / f"stream_{policy_name}.parquet")
    info = json.loads((config.RESULTS_DIR / "runs" / f"{policy_name}.json").read_text())
    return dict(group=group, variant=variant, policy=label, spec=policy_name, **summarise(res, stream, info))


def main(n_trials=8, force=False):
    config.ensure_dirs()
    raw = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(raw)
    rows = []

    best_file = config.RESULTS_DIR / "best_params.json"
    if best_file.exists() and not force:
        best = json.loads(best_file.read_text())
    else:
        best, _ = tune(df, cols, info, n_trials)

    rows += [default_row("class_weight", "with weights (default)", "no_retrain", "no_retrain"),
             default_row("class_weight", "with weights (default)", "fixed 4 wk", "fixed_4w")]
    for label, spec in (("no_retrain", "no_retrain"), ("fixed 4 wk", "fixed:4")):
        rows.append(run_one("class_weight", "no weights", label, spec, df, cols, info, class_weight=False, force=force))

    rows += [default_row("tuning", "default params", "no_retrain", "no_retrain"),
             default_row("tuning", "default params", "fixed 4 wk", "fixed_4w")]
    for label, spec in (("no_retrain", "no_retrain"), ("fixed 4 wk", "fixed:4")):
        rows.append(run_one("tuning", "tuned params", label, spec, df, cols, info, params=best, force=force))

    rows += [default_row("window", "7-day windows (default)", "no_retrain", "no_retrain"),
             default_row("window", "7-day windows (default)", "fixed 4 wk", "fixed_4w"),
             default_row("window", "7-day windows (default)", "adwin", "drift_adwin")]
    df14, cols14, info14 = prepare_data(raw, window_days=14, initial_windows=4)
    for label, spec in (("no_retrain", "no_retrain"), ("fixed 4 wk", "fixed:2"), ("adwin", "drift:adwin")):
        rows.append(run_one("window", "14-day windows", label, spec, df14, cols14, info14, force=force))

    tab = pd.DataFrame(rows)
    base = tab[tab.policy == "no_retrain"].set_index(["group", "variant"])
    tab["pr_auc_gain_pct"] = [100 * (r.mean_pr_auc / base.loc[(r.group, r.variant), "mean_pr_auc"] - 1) for r in tab.itertuples()]
    tab["cost_saving_pct"] = [100 * (base.loc[(r.group, r.variant), "total_cost"] - r.total_cost)
                              / base.loc[(r.group, r.variant), "total_cost"] for r in tab.itertuples()]
    tab.to_csv(config.RESULTS_DIR / "extra_experiments.csv", index=False)
    make_figure(tab)
    pd.set_option("display.width", 220)
    print(tab[["group", "variant", "policy", "mean_pr_auc", "n_retrains", "pr_auc_gain_pct", "cost_saving_pct"]].round(3).to_string(index=False))


def make_figure(tab):
    groups = [("class_weight", "Class weighting"), ("tuning", "Hyper-parameters"), ("window", "Window length")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), sharey=False)
    colors = {"no_retrain": C["red"], "fixed 4 wk": C["blue"], "adwin": C["green"]}
    for ax, (g, title) in zip(axes, groups):
        t = tab[tab.group == g]
        variants = list(dict.fromkeys(t.variant))
        pols = list(dict.fromkeys(t.policy))
        w = 0.8 / len(pols)
        for j, p in enumerate(pols):
            vals = [t[(t.variant == v) & (t.policy == p)].mean_pr_auc.iloc[0] for v in variants]
            ax.bar(np.arange(len(variants)) + j * w, vals, w, label=p, color=colors.get(p, C["gray"]))
        ax.set_xticks(np.arange(len(variants)) + w * (len(pols) - 1) / 2)
        ax.set_xticklabels(variants, fontsize=8)
        ax.set_title(title)
        ax.set_ylabel("Mean weekly PR-AUC")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    viz.save(fig, "extra_experiments")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.trials, a.force)
