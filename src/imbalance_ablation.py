import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE

from src import config, viz
from src.simulation import prepare_data, window_metrics

plt = viz.plt
C = viz.COLORS
T = config.TARGET


def to_numeric_matrix(train, deploy, cols):
    Xtr, Xde = train[cols].copy(), deploy[cols].copy()
    for c in cols:
        if isinstance(Xtr[c].dtype, pd.CategoricalDtype):
            Xtr[c] = Xtr[c].cat.codes.astype("float32")
            Xde[c] = Xde[c].cat.codes.astype("float32")
    med = Xtr.median()
    return Xtr.fillna(med), Xde.fillna(med)


def evaluate(model, deploy, X):
    per = []
    for w, idx in deploy.groupby("window").indices.items():
        p = model.predict_proba(X.iloc[idx])[:, 1]
        per.append(window_metrics(deploy[T].values[idx], p, 0.5))
    per = pd.DataFrame(per)
    return dict(mean_roc_auc=per["roc_auc"].mean(), mean_pr_auc=per["pr_auc"].mean(),
                mean_recall_at_1pct_fpr=per["recall_at_1pct_fpr"].mean())


def main():
    config.ensure_dirs()
    df = pd.read_parquet(config.CLEAN_PARQUET)
    df, cols, info = prepare_data(df)
    iw = info["initial_train_windows"]
    train, deploy = df[df["window"] < iw].reset_index(drop=True), df[df["window"] >= iw].reset_index(drop=True)
    y = train[T].values
    base = {k: v for k, v in config.LGBM_PARAMS.items()}
    Xtr_num, Xde_num = to_numeric_matrix(train, deploy, cols)
    rows = []

    def fit(X, yy, spw=None):
        p = dict(base)
        if spw:
            p["scale_pos_weight"] = spw
        return lgb.LGBMClassifier(**p).fit(X, yy)

    spw = (len(y) - y.sum()) / max(y.sum(), 1)
    rows.append(dict(method="No imbalance handling", **evaluate(fit(Xtr_num, y), deploy, Xde_num)))
    rows.append(dict(method="Class weighting (scale_pos_weight)", **evaluate(fit(Xtr_num, y, spw), deploy, Xde_num)))

    rng = np.random.default_rng(config.SEED)
    pos = np.where(y == 1)[0]
    neg = rng.choice(np.where(y == 0)[0], size=min(10 * len(pos), (y == 0).sum()), replace=False)
    keep = np.sort(np.r_[pos, neg])
    rows.append(dict(method="Random undersampling (10:1)", **evaluate(fit(Xtr_num.iloc[keep], y[keep]), deploy, Xde_num)))
    Xs, ys = SMOTE(random_state=config.SEED, k_neighbors=5).fit_resample(Xtr_num.iloc[keep], y[keep])
    rows.append(dict(method="Undersample 10:1 + SMOTE", **evaluate(fit(Xs, ys), deploy, Xde_num)))

    out = pd.DataFrame(rows)
    out.to_csv(config.TABLE_DIR / "imbalance_ablation.csv", index=False)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    yy = np.arange(len(out))
    ax.barh(yy - 0.2, out["mean_roc_auc"], 0.4, color=C["blue"], label="Mean ROC-AUC")
    ax.barh(yy + 0.2, out["mean_pr_auc"], 0.4, color=C["red"], label="Mean PR-AUC")
    ax.set_yticks(yy); ax.set_yticklabels(out["method"]); ax.invert_yaxis(); ax.legend()
    ax.set_title("Imbalance handling (trained on initial window, scored on all later weeks)")
    viz.save(fig, "imbalance_ablation")
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
