import numpy as np
import pandas as pd

from src import config

NON_FEATURES = {config.ID_COL, config.TIME_COL, config.TARGET}


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df[config.TIME_COL]
    df["hour"] = ((t // 3600) % 24).astype("int8")
    df["dow"] = ((t // config.SECONDS_PER_DAY) % 7).astype("int8")
    if "TransactionAmt" in df:
        df["log_amt"] = np.log1p(df["TransactionAmt"]).astype("float32")
        df["amt_cents"] = ((df["TransactionAmt"] * 1000) % 1000).round().astype("float32")
    return df


def candidate_features(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURES]


def select_features(train_df: pd.DataFrame, cols: list, corr_threshold=config.CORR_THRESHOLD,
                    sample=50_000) -> tuple:
    num = [c for c in cols if pd.api.types.is_numeric_dtype(train_df[c])]
    s = train_df[num]
    if len(s) > sample:
        s = s.sample(sample, random_state=config.SEED)
    corr = s.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    dropped = [c for c in upper.columns if (upper[c] > corr_threshold).any()]
    kept = [c for c in cols if c not in set(dropped)]
    return kept, dropped
