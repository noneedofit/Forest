"""PaySim leakage audit: the same model with and without the balance columns.

python -m eval.leakage_paysim   ->  eval/results/leakage.json
"""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from forest.config import DATA_DIR
from forest.guard.features import FEATURES, FeatureState
from forest.guard.load import load_paysim
from forest.guard.train import fit_boosting, subsample

OUT = Path(__file__).resolve().parent / "results" / "leakage.json"
TRAIN_LAST_STEP = 500
BALANCES = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
RECEIVER = [
    "r_n_senders_24h",
    "r_n_new_senders_24h",
    "r_in_log_24h",
    "r_age_hours",
    "r_out_in_ratio_24h",
]


def receiver_features(df) -> np.ndarray:
    """Streaming receiver features over PaySim (one step = one hour), via the guard's own code."""
    state = FeatureState()
    cols = [FEATURES.index(c) for c in RECEIVER]
    out = np.empty((len(df), len(cols)), dtype=np.float32)
    rows = zip(
        df["step"].tolist(), df["nameOrig"].tolist(), df["nameDest"].tolist(), df["amount"].tolist()
    )
    for i, (step, src, dst, amt) in enumerate(rows):
        f = state.step(
            step * 3600, state.account_id(src), state.account_id(dst), amt, -1, False, False
        )
        out[i] = [f[c] for c in cols]
    return out


def evaluate(x: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray) -> dict:
    tr_s = subsample(train, y)
    model = fit_boosting(x[tr_s], y[tr_s], num_round=300, max_depth=6)
    p = model.predict_proba(x[test])[:, 1]
    return {
        "pr_auc": float(average_precision_score(y[test], p)),
        "roc_auc": float(roc_auc_score(y[test], p)),
    }


def main() -> None:
    df = load_paysim(DATA_DIR / "raw" / "paysim.csv")
    y = df["isFraud"].to_numpy(np.int8)
    amount = df["amount"].to_numpy(np.float32)
    is_transfer = (df["type"] == "TRANSFER").to_numpy(np.float32)
    hour = (df["step"] % 24).to_numpy(np.float32)
    train = np.flatnonzero(df["step"].to_numpy() <= TRAIN_LAST_STEP)
    test = np.flatnonzero(df["step"].to_numpy() > TRAIN_LAST_STEP)

    x_a = np.column_stack([amount, is_transfer, df[BALANCES].to_numpy(np.float32)])
    a = evaluate(x_a, y, train, test)
    del x_a

    x_b = np.column_stack([amount, is_transfer, hour, receiver_features(df)])
    b = evaluate(x_b, y, train, test)

    result = {
        "data": "PaySim, TRANSFER + CASH_OUT only",
        "rows": len(y),
        "split": f"train step <= {TRAIN_LAST_STEP}, test step > {TRAIN_LAST_STEP}",
        "train_positives": int(y[train].sum()),
        "test_positives": int(y[test].sum()),
        "model_a_with_balances": {"features": ["amount", "type", *BALANCES], **a},
        "model_b_no_balances": {"features": ["amount", "type", "hour", *RECEIVER], **b},
        "note": "A looks near perfect because the balance columns encode the label: the fraud "
        "empties the account, and the after-payment balances don't exist yet when we must decide.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(f"A with balances: PR-AUC {a['pr_auc']:.3f}  ROC-AUC {a['roc_auc']:.3f}")
    print(f"B without:       PR-AUC {b['pr_auc']:.3f}  ROC-AUC {b['roc_auc']:.3f}")


if __name__ == "__main__":
    main()
