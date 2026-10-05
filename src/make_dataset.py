import json

import numpy as np
import pandas as pd

from src import config


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for c in df.select_dtypes(include=["float64"]).columns:
        df[c] = df[c].astype("float32")
    for c in df.select_dtypes(include=["int64"]).columns:
        df[c] = pd.to_numeric(df[c], downcast="integer")
    return df


def build(raw_dir=config.RAW_DIR, out=config.CLEAN_PARQUET) -> pd.DataFrame:
    config.ensure_dirs()
    tr = downcast(pd.read_csv(raw_dir / "train_transaction.csv"))
    idn = downcast(pd.read_csv(raw_dir / "train_identity.csv"))
    idn.columns = [c.replace("-", "_") for c in idn.columns]

    report = {
        "transaction_shape": list(tr.shape),
        "identity_shape": list(idn.shape),
        "identity_coverage_pct": round(100 * tr[config.ID_COL].isin(idn[config.ID_COL]).mean(), 2),
    }

    df = tr.merge(idn, on=config.ID_COL, how="left")
    df["has_identity"] = df[config.ID_COL].isin(idn[config.ID_COL]).astype("int8")
    del tr, idn
    report["merged_shape"] = list(df.shape)
    report["memory_mb_after_downcast"] = round(df.memory_usage(deep=True).sum() / 1e6, 1)

    miss = df.isna().mean()
    report["duplicate_rows"] = int(df.duplicated(subset=[c for c in df.columns if c != config.ID_COL]).sum())
    report["duplicate_transaction_ids"] = int(df[config.ID_COL].duplicated().sum())
    report["columns_with_any_missing"] = int((miss > 0).sum())
    report["columns_over_50pct_missing"] = int((miss > 0.5).sum())
    report["columns_over_90pct_missing"] = int((miss > config.MAX_MISSING_FRAC).sum())
    report["overall_missing_pct"] = round(float(df.isna().values.mean() * 100), 2)

    protected = {config.ID_COL, config.TIME_COL, config.TARGET}
    sparse = [c for c in df.columns if miss[c] > config.MAX_MISSING_FRAC and c not in protected]
    constant = [c for c in df.columns if c not in protected and c not in sparse
                and df[c].nunique(dropna=True) <= 1]
    df = df.drop(columns=sparse + constant)
    n_before = len(df)
    df = df.drop_duplicates(subset=config.ID_COL)
    report["dropped_sparse_columns"] = sparse
    report["dropped_constant_columns"] = constant
    report["rows_dropped_duplicate_id"] = n_before - len(df)

    for c in df.select_dtypes(include=["object", "string"]).columns:
        df[c] = df[c].astype("category")

    df = df.sort_values(config.TIME_COL).reset_index(drop=True)

    report["final_shape"] = list(df.shape)
    report["n_fraud"] = int(df[config.TARGET].sum())
    report["fraud_rate_pct"] = round(100 * float(df[config.TARGET].mean()), 3)
    report["time_span_days"] = round(float((df[config.TIME_COL].max() - df[config.TIME_COL].min())
                                           / config.SECONDS_PER_DAY), 2)
    report["n_categorical_cols"] = int(len(df.select_dtypes(include="category").columns))
    report["data_source"] = "synthetic" if "synthetic" in str(raw_dir).lower() else "real"

    df.to_parquet(out, index=False)
    (config.RESULTS_DIR / "data_quality.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if not k.startswith("dropped")}, indent=2))
    print(f"Saved {out}")
    return df


if __name__ == "__main__":
    build()
