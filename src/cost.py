import numpy as np
import pandas as pd

DEFAULTS = dict(review_cost=2.0, retrain_cost=500.0, fraud_loss_rate=1.0)


def components(stream: pd.DataFrame, n_retrains: int) -> dict:
    y, pred, amt = stream["y"].values, stream["pred"].values, stream["amount"].values
    fraud_amt = float((amt * (y == 1)).sum())
    caught = float((amt * ((y == 1) & (pred == 1))).sum())
    return dict(n_retrains=int(n_retrains), fraud_amount=fraud_amt, caught_amount=caught,
                missed_amount=fraud_amt - caught, flagged=int(pred.sum()),
                false_alarms=int(((pred == 1) & (y == 0)).sum()), n_txn=int(len(y)))


def total_cost(c: dict, review_cost=DEFAULTS["review_cost"], retrain_cost=DEFAULTS["retrain_cost"],
               fraud_loss_rate=DEFAULTS["fraud_loss_rate"]) -> dict:
    loss = fraud_loss_rate * c["missed_amount"]
    review = review_cost * c["flagged"]
    retrain = retrain_cost * c["n_retrains"]
    return dict(fraud_loss=loss, review=review, retraining=retrain, total=loss + review + retrain)


def break_even_retrain_cost(c: dict, c_base: dict, review_cost=DEFAULTS["review_cost"],
                            fraud_loss_rate=DEFAULTS["fraud_loss_rate"]):
    if c["n_retrains"] == 0:
        return np.nan
    a = total_cost(c, review_cost, 0.0, fraud_loss_rate)["total"]
    b = total_cost(c_base, review_cost, 0.0, fraud_loss_rate)["total"]
    return (b - a) / c["n_retrains"]
