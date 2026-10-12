import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config, cost, tracking, viz
from src.baseline import decay_statistics
from src.simulation import (CovariateDrift, DriftTriggered, FixedInterval, HybridDrift, NoRetrain, prepare_data,
                            run_simulation)

plt = viz.plt
C = viz.COLORS

DEFAULT_SPECS = ["no_retrain", "fixed:2", "fixed:4", "fixed:8",
                 "drift:adwin_miss", "drift:ddm_miss", "drift:adwin", "drift:ddm", "drift:perf_drop",
                 "fixed:4@7", "fixed:4@14"]

EXTRA_SPECS = ["fixed:4#sliding", "drift:adwin:cd4", "hybrid:adwin:8", "hybrid:ddm_miss:8",
               "drift:psi3", "drift:psi5", "drift:psi8"]


def make_policy(spec: str):
    delay, mode = 0, "expanding"
    if "#sliding" in spec:
        spec, mode = spec.replace("#sliding", ""), "sliding"
    if "@" in spec:
        spec, d = spec.split("@")
        delay = int(d)
    parts = spec.split(":")
    kind = parts[0]
    if kind == "no_retrain":
        pol = NoRetrain()
    elif kind == "fixed":
        pol = FixedInterval(int(parts[1]))
    elif kind == "drift" and parts[1].startswith("psi"):
        pol = CovariateDrift(min_features=int(parts[1][3:]))
    elif kind == "drift":
        cd = int(parts[2][2:]) if len(parts) > 2 and parts[2].startswith("cd") else 2
        pol = DriftTriggered(detector=parts[1], cooldown=cd)
        if cd != 2:
            pol.name = f"{pol.name}_cd{cd}"
    elif kind == "hybrid":
        pol = HybridDrift(detector=parts[1], max_age=int(parts[2]))
    else:
        raise ValueError(spec)
    if mode == "sliding":
        pol.name = f"{pol.name}_sliding"
    if delay:
        pol.name = f"{pol.name}_delay{delay}d"
    return pol, delay, mode


def run_all(specs, force=False):
    config.ensure_dirs()
    runs_dir = config.RESULTS_DIR / "runs"
    runs_dir.mkdir(exist_ok=True)
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(df)
    out = {}
    for spec in specs:
        pol, delay, mode = make_policy(spec)
        jf = runs_dir / f"{pol.name}.json"
        if jf.exists() and not force and (config.TABLE_DIR / f"window_metrics_{pol.name}.csv").exists():
            print(f"[cached] {pol.name}")
            out[pol.name] = json.loads(jf.read_text())
            continue
        res, pinfo, stream, _ = run_simulation(df, cols, pol, delay_days=delay, mode=mode, verbose=False)
        res.to_csv(config.TABLE_DIR / f"window_metrics_{pol.name}.csv", index=False)
        stream.to_parquet(config.RESULTS_DIR / f"stream_{pol.name}.parquet", index=False)
        pinfo["spec"] = spec
        jf.write_text(json.dumps(pinfo, indent=1))
        out[pol.name] = pinfo
        print(f"[done] {pol.name}: retrains={pinfo['n_retrains']} at {pinfo['retrain_windows']} "
              f"meanPR-AUC={res['pr_auc'].mean():.4f} train={pinfo['total_train_seconds']:.0f}s", flush=True)
    return out


def log_to_mlflow(tab: pd.DataFrame, runs: dict):
    for _, r in tab.iterrows():
        tracking.log_policy_run(
            r.policy, params=dict(label_delay_days=r.label_delay_days, n_retrains=r.n_retrains,
                                  retrain_windows=r.retrain_windows, window_days=config.WINDOW_DAYS),
            metrics=dict(mean_pr_auc=r.mean_pr_auc, mean_roc_auc=r.mean_roc_auc, mean_f1=r.mean_f1,
                         mean_recall_at_1pct_fpr=r.mean_recall_at_1pct_fpr, total_cost=r.total_cost,
                         train_seconds=r.train_seconds),
            artifacts=[config.TABLE_DIR / f"window_metrics_{r.policy}.csv"])


def summarise(runs: dict) -> pd.DataFrame:
    base_stream = pd.read_parquet(config.RESULTS_DIR / "stream_no_retrain.parquet")
    base_c = cost.components(base_stream, 0)
    base_w = pd.read_csv(config.TABLE_DIR / "window_metrics_no_retrain.csv").set_index("window")
    rows = []
    for name, info in runs.items():
        w = pd.read_csv(config.TABLE_DIR / f"window_metrics_{name}.csv").set_index("window")
        st = pd.read_parquet(config.RESULTS_DIR / f"stream_{name}.parquet")
        comp = cost.components(st, info["n_retrains"])
        tc = cost.total_cost(comp)
        common = w.index.intersection(base_w.index)
        diff = (w.loc[common, "pr_auc"] - base_w.loc[common, "pr_auc"]).dropna()
        try:
            p = stats.wilcoxon(diff).pvalue if len(diff) > 5 and (diff != 0).any() else np.nan
        except ValueError:
            p = np.nan
        rows.append(dict(
            policy=name, label_delay_days=info["label_delay_days"], n_retrains=info["n_retrains"],
            retrain_windows=" ".join(map(str, info["retrain_windows"])),
            train_seconds=round(info["total_train_seconds"], 1),
            mean_roc_auc=w["roc_auc"].mean(), mean_pr_auc=w["pr_auc"].mean(),
            mean_recall_at_1pct_fpr=w["recall_at_1pct_fpr"].mean(), mean_f1=w["f1"].mean(),
            worst_week_pr_auc=w["pr_auc"].min(),
            pr_auc_gain_vs_no_retrain_pct=100 * (w["pr_auc"].mean() / base_w["pr_auc"].mean() - 1),
            wilcoxon_p_vs_no_retrain=p,
            missed_fraud_amount=comp["missed_amount"], flagged=comp["flagged"],
            false_alarms=comp["false_alarms"], fraud_loss_cost=tc["fraud_loss"], review_cost=tc["review"],
            retraining_cost=tc["retraining"], total_cost=tc["total"],
            break_even_retrain_cost=cost.break_even_retrain_cost(comp, base_c)))
    out = pd.DataFrame(rows)
    nr = out.loc[out.policy == "no_retrain", "total_cost"].iloc[0]
    out["cost_saving_vs_no_retrain_pct"] = 100 * (nr - out["total_cost"]) / nr
    return out


def make_figures(tab: pd.DataFrame, runs: dict):
    main = tab[tab.label_delay_days == 0].copy()
    drift = main[main.policy.str.startswith(("drift_", "hybrid_"))]
    best_drift = drift.sort_values("mean_pr_auc", ascending=False).policy.iloc[0] if len(drift) else None
    show = [p for p in ("no_retrain", "fixed_4w", best_drift) if p]
    colors = {"no_retrain": C["red"], "fixed_4w": C["blue"], best_drift: C["green"]}

    fig, ax = plt.subplots(figsize=(10, 4))
    for p in show:
        w = pd.read_csv(config.TABLE_DIR / f"window_metrics_{p}.csv")
        ax.plot(w["window"], w["pr_auc"], marker="o", ms=3, color=colors[p], label=p)
        for rw in runs[p]["retrain_windows"]:
            ax.axvline(rw, color=colors[p], ls=":", alpha=0.5)
    ax.set_xlabel("Deployment week index"); ax.set_ylabel("PR-AUC")
    ax.set_title("Weekly PR-AUC by policy (dotted lines = retraining)"); ax.legend()
    viz.save(fig, "policy_timeline")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for _, r in main.iterrows():
        col = C["green"] if r.policy.startswith(("drift_", "hybrid_")) else C["blue"] if r.policy.startswith("fixed") else C["red"]
        ax[0].scatter(r.n_retrains, r.mean_pr_auc, s=70, color=col)
        ax[0].annotate(r.policy.replace("drift_", "d:"), (r.n_retrains, r.mean_pr_auc), fontsize=8,
                       xytext=(4, 4), textcoords="offset points")
    ax[0].set_xlabel("Number of retrainings"); ax[0].set_ylabel("Mean weekly PR-AUC")
    ax[0].set_title("Performance vs retraining effort (blue fixed, green drift-triggered)")
    m2 = main.sort_values("total_cost")
    bottom = np.zeros(len(m2))
    for col, lab, colr in (("fraud_loss_cost", "Missed fraud", C["red"]), ("review_cost", "Review", C["orange"]),
                           ("retraining_cost", "Retraining", C["blue"])):
        ax[1].barh(m2.policy, m2[col], left=bottom, color=colr, label=lab)
        bottom += m2[col].values
    ax[1].invert_yaxis(); ax[1].legend(); ax[1].set_xlabel("Cost (default constants)")
    ax[1].set_title(f"Total cost (review ${cost.DEFAULTS['review_cost']:.0f}, retrain ${cost.DEFAULTS['retrain_cost']:.0f})")
    fig.tight_layout(); viz.save(fig, "policy_tradeoff")

    comps = {}
    for p in main.policy:
        st = pd.read_parquet(config.RESULTS_DIR / f"stream_{p}.parquet")
        comps[p] = cost.components(st, runs[p]["n_retrains"])
    rc = np.geomspace(10, 100_000, 25)
    rv = np.array([0.5, 1, 2, 5, 10, 20])
    names = list(comps)
    grid = np.zeros((len(rv), len(rc)), dtype=int)
    for i, r in enumerate(rv):
        for j, k in enumerate(rc):
            grid[i, j] = int(np.argmin([cost.total_cost(comps[p], r, k)["total"] for p in names]))
    fig, ax = plt.subplots(figsize=(9, 3.8))
    im = ax.imshow(grid, aspect="auto", cmap="tab20", vmin=0, vmax=19, origin="lower")
    ax.set_xticks(range(0, len(rc), 4)); ax.set_xticklabels([f"{rc[k]:,.0f}" for k in range(0, len(rc), 4)])
    ax.set_yticks(range(len(rv))); ax.set_yticklabels([f"${r}" for r in rv]); ax.grid(False)
    ax.set_xlabel("Cost of ONE retraining ($)"); ax.set_ylabel("Review cost per flagged txn")
    used = sorted(set(grid.ravel()))
    ax.set_title("Cheapest policy under different cost assumptions")
    handles = [plt.Rectangle((0, 0), 1, 1, color=plt.get_cmap("tab20")(u / 19)) for u in used]
    ax.legend(handles, [names[u] for u in used], bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    viz.save(fig, "cost_sensitivity")
    return best_drift


def main(specs=None, force=False):
    runs = run_all(specs or (DEFAULT_SPECS + EXTRA_SPECS), force)
    tab = summarise(runs)
    tab.to_csv(config.RESULTS_DIR / "policy_comparison.csv", index=False)
    log_to_mlflow(tab, runs)
    best_drift = make_figures(tab, runs)
    main_tab = tab[tab.label_delay_days == 0]
    summary = dict(cost_defaults=cost.DEFAULTS, best_drift_policy=best_drift,
                   best_by_pr_auc=main_tab.sort_values("mean_pr_auc", ascending=False).policy.iloc[0],
                   cheapest_at_defaults=main_tab.sort_values("total_cost").policy.iloc[0],
                   policies=tab.round(4).to_dict("records"))
    (config.RESULTS_DIR / "policy_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(tab[["policy", "n_retrains", "mean_pr_auc", "pr_auc_gain_vs_no_retrain_pct", "wilcoxon_p_vs_no_retrain",
               "total_cost", "cost_saving_vs_no_retrain_pct", "break_even_retrain_cost"]].round(3).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="+")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    main(a.only, a.force)
