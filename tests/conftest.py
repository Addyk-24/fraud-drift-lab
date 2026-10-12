import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
_tmp = tempfile.mkdtemp(prefix="fraud_test_")
os.environ["FRAUD_DATA_DIR"] = os.path.join(_tmp, "data")
os.environ["FRAUD_RESULTS_DIR"] = os.path.join(_tmp, "results")

import pytest

from src import config, make_dataset, synthetic


@pytest.fixture(scope="session")
def clean_df():
    config.ensure_dirs()
    tr, idn = synthetic.generate(n=14_000, seed=1)
    tr.to_csv(config.RAW_DIR / "train_transaction.csv", index=False)
    idn.to_csv(config.RAW_DIR / "train_identity.csv", index=False)
    return make_dataset.build(raw_dir=config.RAW_DIR)


@pytest.fixture(scope="session")
def prepared(clean_df):
    from src.simulation import prepare_data
    return prepare_data(clean_df)
