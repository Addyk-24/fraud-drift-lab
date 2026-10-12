import numpy as np
import pandas as pd
from scipy import stats


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
