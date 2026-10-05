import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def generate(n=60_000, days=183, seed=0):
    rng = np.random.default_rng(seed)
    t = np.sort(rng.integers(86_400, days * 86_400, n))
    day = t / 86_400
    a = np.clip((day - 90) / 30, 0, 1)

    tid = 2_987_000 + np.arange(n)
    amt = np.round(np.exp(rng.normal(4.4, 1.2, n)), 2)
    product = rng.choice(list("WCHRS"), n, p=[0.74, 0.10, 0.07, 0.06, 0.03])
    card4 = rng.choice(["visa", "mastercard", "discover", "american express"], n, p=[0.65, 0.3, 0.02, 0.03])
    card6 = rng.choice(["debit", "credit"], n, p=[0.75, 0.25])
    email = rng.choice(["gmail.com", "yahoo.com", "hotmail.com", "anonymous.com", "protonmail.com"],
                       n, p=[0.45, 0.25, 0.15, 0.1, 0.05])
    C = rng.poisson(2, (n, 6)).astype("float32")
    D = rng.exponential(30, (n, 5)).astype("float32")
    D[rng.random((n, 5)) < 0.45] = np.nan
    blocks = rng.normal(size=(n, 6))
    V = np.hstack([blocks[:, [i]] + 0.2 * rng.normal(size=(n, 5)) for i in range(6)]).astype("float32")
    V[rng.random(V.shape) < 0.3] = np.nan

    zA = 0.6 * (np.log(amt) - 4.4) + 0.5 * (C[:, 0] - 2) + 1.2 * (product == "C")
    zB = 1.4 * (email == "protonmail.com") + 0.9 * blocks[:, 0] - 0.02 * np.nan_to_num(D[:, 0], nan=30) \
        + 0.8 * (product == "C") * 0
    logit = -4.3 + (1 - a) * zA + a * zB + 0.3 * (card6 == "credit") + 0.4 * rng.normal(size=n)
    p = 1 / (1 + np.exp(-logit))
    fraud = (rng.random(n) < p).astype("int8")

    tr = pd.DataFrame({
        "TransactionID": tid, "isFraud": fraud, "TransactionDT": t, "TransactionAmt": amt,
        "ProductCD": product, "card1": rng.integers(1000, 18000, n), "card2": rng.integers(100, 600, n).astype("float32"),
        "card3": rng.choice([150.0, 185.0, 106.0], n), "card4": card4, "card5": rng.choice([226.0, 224.0, 166.0], n),
        "card6": card6, "addr1": rng.integers(100, 540, n).astype("float32"),
        "addr2": np.where(rng.random(n) < 0.98, 87.0, 60.0).astype("float32"),
        "P_emaildomain": email, "R_emaildomain": np.where(rng.random(n) < 0.6, None, email),
    })
    for i in range(6):
        tr[f"C{i + 1}"] = C[:, i]
    for i in range(5):
        tr[f"D{i + 1}"] = D[:, i]
    for i in range(9):
        m = rng.choice(["T", "F"], n)
        tr[f"M{i + 1}"] = np.where(rng.random(n) < 0.4, None, m)
    for i in range(V.shape[1]):
        tr[f"V{i + 1}"] = V[:, i]
    tr["dead_col"] = np.where(rng.random(n) < 0.97, np.nan, 1.0)

    has_id = rng.random(n) < 0.24
    idn = pd.DataFrame({
        "TransactionID": tid[has_id],
        "id_01": rng.integers(-100, 0, has_id.sum()).astype("float32"),
        "id_12": rng.choice(["Found", "NotFound"], has_id.sum()),
        "id_30": rng.choice(["Windows 10", "iOS 11.1.0", "Android 7.0", "Mac OS X 10_12_6"], has_id.sum()),
        "id_38": rng.choice(["T", "F"], has_id.sum()),
        "DeviceType": rng.choice(["desktop", "mobile"], has_id.sum()),
        "DeviceInfo": rng.choice(["Windows", "iOS Device", "SM-G960F", "MacOS"], has_id.sum()),
    })
    return tr, idn


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data_synthetic/raw")
    ap.add_argument("--n", type=int, default=60_000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tr, idn = generate(args.n)
    tr.to_csv(out / "train_transaction.csv", index=False)
    idn.to_csv(out / "train_identity.csv", index=False)
    print(f"wrote {len(tr)} transactions, fraud rate {tr.isFraud.mean():.3%} -> {out}")
