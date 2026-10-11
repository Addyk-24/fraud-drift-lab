import json

import numpy as np
import pandas as pd
from scipy import stats

from src import config, viz
from src.simulation import prepare_data, train_with_threshold, training_slice

plt = viz.plt
C = viz.COLORS


def contributions(model, X):
    c = model.predict(X, pred_contrib=True)
    return c[:, :-1]


def main(n_sample=4000, top_k=15):
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(df)
    iw = info["initial_train_windows"]
    model, _ = train_with_threshold(training_slice(df, iw, 0), cols)
    rng = np.random.default_rng(config.SEED)

    weekly, late_X, late_S = {}, [], []
    weeks = sorted(w for w in df["window"].unique() if w >= iw)
    for w in weeks:
        cur = df[df["window"] == w]
        fr = cur[cur[config.TARGET] == 1]
        fr = fr.sample(min(len(fr), n_sample // 4), random_state=config.SEED)
        ok = cur[cur[config.TARGET] == 0].sample(min(n_sample - len(fr), (cur[config.TARGET] == 0).sum()),
                                                 random_state=config.SEED)
        s = pd.concat([fr, ok])
        sv = contributions(model, s[cols])
        weekly[w] = np.abs(sv).mean(axis=0)
        if w >= weeks[-3]:
            late_X.append(s[cols]); late_S.append(sv)
    byweek = pd.DataFrame(weekly, index=cols).T
    byweek.index.name = "window"
    byweek.to_csv(config.TABLE_DIR / "shap_mean_abs_by_week.csv")

    early = byweek.loc[weeks[:3]].mean()
    late = byweek.loc[weeks[-3:]].mean()
    tab = pd.DataFrame({"early_mean_abs_shap": early, "late_mean_abs_shap": late})
    tab["early_share_pct"] = 100 * tab["early_mean_abs_shap"] / tab["early_mean_abs_shap"].sum()
    tab["late_share_pct"] = 100 * tab["late_mean_abs_shap"] / tab["late_mean_abs_shap"].sum()
    tab["rank_early"] = tab["early_mean_abs_shap"].rank(ascending=False).astype(int)
    tab["rank_late"] = tab["late_mean_abs_shap"].rank(ascending=False).astype(int)
    tab["rank_change"] = tab["rank_early"] - tab["rank_late"]
    tab = tab.sort_values("early_mean_abs_shap", ascending=False)
    tab.index.name = "feature"
    tab.to_csv(config.TABLE_DIR / "shap_early_vs_late.csv")

    rho, p = stats.spearmanr(tab["early_mean_abs_shap"], tab["late_mean_abs_shap"])
    top10e, top10l = set(early.nlargest(10).index), set(late.nlargest(10).index)
    movers = tab[(tab["rank_early"] <= 40) | (tab["rank_late"] <= 40)].sort_values("rank_change")
    summary = dict(
        spearman_rank_corr_early_vs_late=float(rho), spearman_p=float(p),
        top10_overlap=len(top10e & top10l), top10_early=list(early.nlargest(10).index),
        top10_late=list(late.nlargest(10).index),
        risers=movers.tail(5)["rank_change"].to_dict(), fallers=movers.head(5)["rank_change"].to_dict(),
        top5_share_pct_early=float(tab["early_share_pct"].head(5).sum()),
        top5_share_pct_late=float(tab.sort_values("late_share_pct", ascending=False)["late_share_pct"].head(5).sum()))
    (config.RESULTS_DIR / "shap_summary.json").write_text(json.dumps(summary, indent=1, default=float))

    top = tab.sort_values("early_mean_abs_shap", ascending=False).head(top_k).index.tolist()
    fig, ax = plt.subplots(figsize=(7, 5.5))
    y = np.arange(len(top))[::-1]
    ax.barh(y + 0.2, tab.loc[top, "early_mean_abs_shap"], 0.4, color=C["blue"], label="First 3 deployment weeks")
    ax.barh(y - 0.2, tab.loc[top, "late_mean_abs_shap"], 0.4, color=C["red"], label="Last 3 deployment weeks")
    ax.set_yticks(y); ax.set_yticklabels(top); ax.set_xlabel("Mean |SHAP value| (log-odds)")
    ax.set_title("What the deployed model relies on, early vs late"); ax.legend()
    viz.save(fig, "shap_global_top15")

    share = byweek[top].div(byweek.sum(axis=1), axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(share.T.values, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top, fontsize=8)
    ax.set_xticks(range(0, len(weeks), 2)); ax.set_xticklabels(weeks[::2]); ax.grid(False)
    ax.set_xlabel("Deployment week index"); ax.set_title("Share of total |SHAP| per feature over time")
    fig.colorbar(im, ax=ax, label="% of total")
    viz.save(fig, "shap_weekly_heatmap")

    X = pd.concat(late_X); S = np.vstack(late_S)
    order = list(pd.Series(np.abs(S).mean(0), index=cols).nlargest(12).index)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for r, f in enumerate(order[::-1]):
        j = cols.index(f)
        v = X[f]
        v = v.cat.codes.astype(float) if isinstance(v.dtype, pd.CategoricalDtype) else v.astype(float)
        pct = v.rank(pct=True).fillna(0.5).values
        idx = rng.choice(len(S), min(1500, len(S)), replace=False)
        ax.scatter(S[idx, j], r + rng.uniform(-0.3, 0.3, len(idx)), c=pct[idx], cmap="coolwarm", s=5, alpha=0.7)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order[::-1]); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("SHAP value (impact on fraud log-odds)")
    ax.set_title("Feature effects in the last 3 weeks (blue = low value, red = high)")
    viz.save(fig, "shap_beeswarm_late")
    print(json.dumps(summary, indent=1, default=float))


if __name__ == "__main__":
    main()
