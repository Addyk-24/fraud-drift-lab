import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = Path(os.environ.get("FRAUD_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
RESULTS_DIR = Path(os.environ.get("FRAUD_RESULTS_DIR", ROOT / "results"))
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"

CLEAN_PARQUET = PROCESSED_DIR / "ieee_cis_clean.parquet"

TARGET = "isFraud"
TIME_COL = "TransactionDT"
ID_COL = "TransactionID"
SECONDS_PER_DAY = 86_400

MAX_MISSING_FRAC = 0.90
CORR_THRESHOLD = 0.95

WINDOW_DAYS = 7
INITIAL_TRAIN_WINDOWS = 8
LABEL_DELAY_DAYS = 0
SEED = 42

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=64,
    colsample_bytree=0.5,
    subsample=0.8,
    subsample_freq=1,
    min_child_samples=50,
    reg_lambda=1.0,
    n_jobs=-1,
    random_state=SEED,
    verbose=-1,
)


def ensure_dirs():
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR, FIG_DIR, TABLE_DIR):
        d.mkdir(parents=True, exist_ok=True)
