import json

import numpy as np
import pandas as pd

from src import config, viz

plt = viz.plt
C = viz.COLORS
T = config.TARGET


def weekly(df):
    t0 = df[config.TIME_COL].min()
    return ((df[config.TIME_COL] - t0) // (7 * config.SECONDS_PER_DAY)).astype(int)


def rate_by(df, col, top=8, min_n=200):
    g = df.groupby(col, observed=True)[T].agg(["mean", "size"])
    g = g[g["size"] >= min_n].sort_values("size", ascending=False).head(top)
    return g


def main():
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df["week"] = weekly(df)
    stats = {}

    n, nf = len(df), int(df[T].sum())
    stats.update(n_transactions=n, n_fraud=nf, fraud_rate_pct=round(100 * nf / n, 3),
                 imbalance_ratio=round((n - nf) / nf, 1))
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.bar(["Legitimate", "Fraud"], [n - nf, nf], color=[C["blue"], C["red"]])
    for i, v in enumerate([n - nf, nf]):
        ax.text(i, v, f"{v:,}\n({100 * v / n:.2f}%)", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, n * 1.15); ax.set_title("Severe class imbalance"); ax.set_ylabel("Transactions")
    viz.save(fig, "eda_class_balance")

    wk = df.groupby("week")[T].agg(["size", "mean"])
    stats["weekly_fraud_rate_pct_min"] = round(100 * wk["mean"].min(), 3)
    stats["weekly_fraud_rate_pct_max"] = round(100 * wk["mean"].max(), 3)
    stats["n_weeks"] = int(len(wk))
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.bar(wk.index, wk["size"], color=C["blue"], alpha=0.35, label="Transactions / week")
    ax.set_xlabel("Week index"); ax.set_ylabel("Transactions")
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(wk.index, 100 * wk["mean"], color=C["red"], marker="o", ms=3, label="Fraud rate (%)")
    ax2.set_ylabel("Fraud rate (%)")
    ax.set_title("Transaction volume and fraud rate per week")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    viz.save(fig, "eda_fraud_over_time")

    miss = (df.isna().mean() * 100).sort_values(ascending=False)
    stats["columns_total"] = int(df.shape[1])
    stats["columns_any_missing"] = int((miss > 0).sum())
    stats["columns_over_50pct_missing"] = int((miss > 50).sum())
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    ax[0].hist(miss, bins=20, color=C["gray"]); ax[0].set_xlabel("% missing"); ax[0].set_ylabel("Columns")
    ax[0].set_title("Missingness across columns (after cleaning)")
    t = miss.head(20).iloc[::-1]
    ax[1].barh(t.index, t.values, color=C["orange"]); ax[1].set_xlabel("% missing")
    ax[1].set_title("20 most incomplete columns"); ax[1].tick_params(axis="y", labelsize=8)
    fig.tight_layout(); viz.save(fig, "eda_missingness")

    cats = [c for c in ("ProductCD", "card4", "card6", "DeviceType", "P_emaildomain", "has_identity") if c in df]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, c in zip(axes.ravel(), cats):
        g = rate_by(df, c)
        ax.bar(g.index.astype(str), 100 * g["mean"], color=C["red"], alpha=0.8)
        ax.axhline(100 * nf / n, color="black", ls="--", lw=1)
        ax.set_title(f"Fraud rate by {c}"); ax.set_ylabel("%"); ax.tick_params(axis="x", rotation=45, labelsize=8)
    for ax in axes.ravel()[len(cats):]:
        ax.axis("off")
    fig.tight_layout(); viz.save(fig, "eda_fraud_by_category")
    if "ProductCD" in df:
        g = rate_by(df, "ProductCD")
        stats["fraud_rate_by_ProductCD_pct"] = (100 * g["mean"]).round(2).to_dict()
    if "has_identity" in df:
        stats["fraud_rate_with_identity_pct"] = round(100 * df.loc[df.has_identity == 1, T].mean(), 2)
        stats["fraud_rate_without_identity_pct"] = round(100 * df.loc[df.has_identity == 0, T].mean(), 2)

    if "TransactionAmt" in df:
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
        for v, col, lab in ((0, C["blue"], "Legitimate"), (1, C["red"], "Fraud")):
            ax[0].hist(np.log10(df.loc[df[T] == v, "TransactionAmt"].clip(lower=0.01)), bins=50,
                       density=True, alpha=0.55, color=col, label=lab)
        ax[0].set_xlabel("log10(TransactionAmt)"); ax[0].set_ylabel("Density"); ax[0].legend()
        ax[0].set_title("Amount distribution by class")
        data = [df.loc[df[T] == 0, "TransactionAmt"], df.loc[df[T] == 1, "TransactionAmt"]]
        ax[1].boxplot(data, tick_labels=["Legitimate", "Fraud"], showfliers=False)
        ax[1].set_title("Transaction amount (outliers hidden)"); ax[1].set_ylabel("Amount")
        fig.tight_layout(); viz.save(fig, "eda_amount")
        stats["median_amount_legit"] = round(float(data[0].median()), 2)
        stats["median_amount_fraud"] = round(float(data[1].median()), 2)
        stats["mean_amount_legit"] = round(float(data[0].mean()), 2)
        stats["mean_amount_fraud"] = round(float(data[1].mean()), 2)

    hr = ((df[config.TIME_COL] // 3600) % 24).astype(int)
    g = df.groupby(hr)[T].agg(["mean", "size"])
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(g.index, g["size"], color=C["blue"], alpha=0.3); ax.set_ylabel("Transactions")
    ax2 = ax.twinx(); ax2.grid(False); ax2.spines["right"].set_visible(True)
    ax2.plot(g.index, 100 * g["mean"], color=C["red"], marker="o", ms=3); ax2.set_ylabel("Fraud rate (%)")
    ax.set_xlabel("Hour of day (relative to dataset reference)"); ax.set_title("Daily cycle")
    viz.save(fig, "eda_hourly")
    stats["peak_fraud_hour"] = int(g["mean"].idxmax())

    num = df.select_dtypes(include="number").drop(columns=[T, config.ID_COL, config.TIME_COL, "week"], errors="ignore")
    s = df[[T]].join(num).sample(min(200_000, len(df)), random_state=config.SEED)
    corr = s.corr()[T].drop(T).dropna()
    top = corr.reindex(corr.abs().sort_values(ascending=False).index).head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.barh(top.index, top.values, color=[C["red"] if v > 0 else C["blue"] for v in top.values])
    ax.set_xlabel(f"Pearson correlation with {T}"); ax.set_title("Top-20 numeric features by |correlation|")
    viz.save(fig, "eda_target_correlation")
    stats["top_correlated_features"] = {k: round(float(v), 3) for k, v in top.iloc[::-1].head(5).items()}

    panels = [c for c in ("ProductCD", "P_emaildomain") if c in df]
    if panels:
        fig, axes = plt.subplots(1, len(panels), figsize=(6 * len(panels), 3.8), squeeze=False)
        for ax, c in zip(axes[0], panels):
            top_cats = df[c].value_counts().head(5).index
            piv = (df[df[c].isin(top_cats)].groupby(["week", c], observed=True)[T].mean().unstack() * 100)
            for col in piv.columns:
                ax.plot(piv.index, piv[col].rolling(2, min_periods=1).mean(), marker="o", ms=2.5, label=str(col))
            ax.set_xlabel("Week index"); ax.set_ylabel("Fraud rate (%)")
            ax.set_title(f"Fraud rate over time by {c}"); ax.legend(fontsize=8)
        fig.tight_layout(); viz.save(fig, "eda_pattern_shift")
    if "TransactionAmt" in df:
        fz = df[df[T] == 1].groupby("week")["TransactionAmt"].median()
        lz = df[df[T] == 0].groupby("week")["TransactionAmt"].median()
        fig, ax = plt.subplots(figsize=(8, 3.4))
        ax.plot(fz.index, fz.values, color=C["red"], marker="o", ms=3, label="Fraud")
        ax.plot(lz.index, lz.values, color=C["blue"], marker="o", ms=3, label="Legitimate")
        ax.set_xlabel("Week index"); ax.set_ylabel("Median amount"); ax.legend()
        ax.set_title("Median transaction amount per week")
        viz.save(fig, "eda_amount_over_time")

    (config.RESULTS_DIR / "eda_stats.json").write_text(json.dumps(stats, indent=2, default=float))
    print(json.dumps(stats, indent=2, default=float))


if __name__ == "__main__":
    main()
