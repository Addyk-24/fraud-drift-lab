import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config, viz

plt = viz.plt
C = viz.COLORS
N_BOOT = 5000
BLOCK = 3


def weekly(name):
    return pd.read_csv(config.TABLE_DIR / f"window_metrics_{name}.csv").set_index("window")["pr_auc"]


def block_bootstrap_ci(diff, rng, block=BLOCK, n_boot=N_BOOT):
    d = np.asarray(diff, dtype=float)
    n = len(d)
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = np.concatenate([np.arange(s, s + block) for s in rng.choice(starts, n_blocks)])[:n]
        means[i] = d[idx].mean()
    return np.percentile(means, [2.5, 97.5])


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])
        adj[i] = min(1.0, running)
    return adj


def compare(a_name, b_name, rng):
    a, b = weekly(a_name), weekly(b_name)
    common = a.index.intersection(b.index)
    diff = (a.loc[common] - b.loc[common]).dropna()
    try:
        p = stats.wilcoxon(diff).pvalue if (diff != 0).any() else 1.0
    except ValueError:
        p = np.nan
    lo, hi = block_bootstrap_ci(diff.values, rng)
    return dict(mean_diff=float(diff.mean()), ci_low=float(lo), ci_high=float(hi), wilcoxon_p=float(p),
                weeks_better=int((diff > 0).sum()), n_weeks=int(len(diff)))


def main():
    config.ensure_dirs()
    rng = np.random.default_rng(config.SEED)
    pc = pd.read_csv(config.RESULTS_DIR / "policy_comparison.csv")
    main_tab = pc[pc.label_delay_days == 0]
    names = [p for p in main_tab.policy if p != "no_retrain"]

    rows = []
    for n in names:
        r = compare(n, "no_retrain", rng)
        r.update(policy=n, n_retrains=int(main_tab.set_index("policy").loc[n, "n_retrains"]))
        rows.append(r)
    vs_base = pd.DataFrame(rows)
    vs_base["wilcoxon_p_holm"] = holm(vs_base["wilcoxon_p"].fillna(1.0))
    vs_base["significant_holm_5pct"] = vs_base["wilcoxon_p_holm"] < 0.05
    vs_base.to_csv(config.RESULTS_DIR / "policy_stats_vs_no_retrain.csv", index=False)

    mt = main_tab.set_index("policy")
    fixed = [p for p in mt.index if p.startswith("fixed_") and "delay" not in p and "sliding" not in p]
    triggered = [p for p in mt.index if p.startswith(("drift_", "hybrid_"))]
    pair_rows = []
    for t in triggered:
        for f in fixed:
            if mt.loc[t, "n_retrains"] == mt.loc[f, "n_retrains"] and mt.loc[t, "n_retrains"] > 0:
                r = compare(t, f, rng)
                r.update(triggered=t, fixed=f, n_retrains=int(mt.loc[t, "n_retrains"]))
                pair_rows.append(r)
    pairs = pd.DataFrame(pair_rows)
    if len(pairs):
        pairs["wilcoxon_p_holm"] = holm(pairs["wilcoxon_p"].fillna(1.0))
    pairs.to_csv(config.RESULTS_DIR / "policy_stats_matched_budget.csv", index=False)

    v = vs_base.sort_values("mean_diff")
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(v) + 1.4))
    y = np.arange(len(v))
    colors = [C["green"] if p.startswith(("drift_", "hybrid_")) else C["blue"] for p in v.policy]
    ax.errorbar(v.mean_diff, y, xerr=[v.mean_diff - v.ci_low, v.ci_high - v.mean_diff], fmt="none", ecolor=C["gray"], capsize=3)
    ax.scatter(v.mean_diff, y, c=colors, zorder=3)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(v.policy, fontsize=8)
    ax.set_xlabel("Mean weekly PR-AUC gain over never retraining (95% block-bootstrap CI)")
    ax.set_title("Policy gains with uncertainty (blue fixed, green drift-triggered)")
    viz.save(fig, "policy_forest")

    summary = dict(n_boot=N_BOOT, block=BLOCK,
                   all_significant_after_holm=bool(vs_base["significant_holm_5pct"].all()),
                   n_policies=int(len(vs_base)),
                   n_significant=int(vs_base["significant_holm_5pct"].sum()),
                   matched_budget_pairs=pairs.round(4).to_dict("records") if len(pairs) else [])
    (config.RESULTS_DIR / "policy_stats_summary.json").write_text(json.dumps(summary, indent=1, default=float))
    pd.set_option("display.width", 220)
    print(vs_base[["policy", "n_retrains", "mean_diff", "ci_low", "ci_high", "wilcoxon_p_holm"]].round(4).to_string(index=False))
    print(pairs.round(4).to_string(index=False) if len(pairs) else "no matched-budget pairs")


if __name__ == "__main__":
    main()
