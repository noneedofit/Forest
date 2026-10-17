"""Build streaming features over IBM AML HI-Small, train the Snap ML guard model, pick thresholds.

python -m forest.guard.train                 # features (cached) + grid + artefacts
python -m forest.guard.train --limit 500000  # feature pass only, to check speed and memory
"""

import argparse
import json
import pickle
import resource
import sys
import time

import numpy as np
from sklearn.metrics import average_precision_score
from snapml import BoostingMachineClassifier

from forest.config import DATA_DIR
from forest.guard.features import FEATURES, FeatureState, featurise_rows, prepare_aml, to_epoch
from forest.guard.load import load_aml, write_fx_rates
from forest.guard.score import load_model

RAW = DATA_DIR / "raw" / "HI-Small_Trans.csv"
FEATURE_DIR = DATA_DIR / "features"
ARTIFACTS = DATA_DIR / "artifacts"

# HI-Small's normal traffic stops after Sep 10; Sep 11-18 is ~1.1k rows, mostly laundering
# chains finishing, so it's featurised (for state) but left out of train/val/test.
TRAIN_END = to_epoch("2022/09/07 00:00")
VAL_END = to_epoch("2022/09/09 00:00")
TEST_END = to_epoch("2022/09/11 00:00")
PAUSE_BUDGET = 0.01
HOLD_BUDGET = 0.001
NEG_PER_POS = 200
GRID = [(300, 4), (300, 6), (300, 8), (500, 4), (500, 6), (500, 8)]


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def build_features(limit: int | None = None) -> None:
    """One time-ordered pass. Snapshots the feature state at the end of validation so the
    scorer starts the test period with real history."""
    t0 = time.time()
    df = load_aml(RAW, nrows=limit)
    print(f"loaded {len(df):,} rows in {time.time() - t0:.0f}s, rss {peak_rss_mb():,.0f} MB")
    state = FeatureState()
    cols = prepare_aml(df, state)
    y = df["y"].to_numpy(np.int8)
    ts = df["ts"].to_numpy(np.int64)
    del df
    print(f"prepared {len(state.ids):,} accounts, rss {peak_rss_mb():,.0f} MB")

    k = int(np.searchsorted(ts, VAL_END))
    t1 = time.time()
    x_head = featurise_rows(state, cols, 0, k)
    if limit is None:
        ARTIFACTS.mkdir(parents=True, exist_ok=True)
        with open(ARTIFACTS / "feature_state.pkl", "wb") as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    x_tail = featurise_rows(state, cols, k, len(ts))
    x = np.concatenate([x_head, x_tail])
    secs = time.time() - t1
    print(f"featurised {len(x):,} rows in {secs:.0f}s ({len(x) / secs:,.0f} rows/s)")
    print(f"peak rss {peak_rss_mb():,.0f} MB")
    if limit is not None:
        return
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FEATURE_DIR / "aml_features.npy", x)
    np.save(FEATURE_DIR / "aml_labels.npy", y)
    np.save(FEATURE_DIR / "aml_ts.npy", ts)


def load_features() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.load(FEATURE_DIR / "aml_features.npy")
    y = np.load(FEATURE_DIR / "aml_labels.npy")
    ts = np.load(FEATURE_DIR / "aml_ts.npy")
    return x, y, ts


def split_index(ts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Row indices for train (Sep 1-6), val (Sep 7-8) and test (Sep 9-10)."""
    a, b, c = (int(np.searchsorted(ts, t)) for t in (TRAIN_END, VAL_END, TEST_END))
    return np.arange(0, a), np.arange(a, b), np.arange(b, c)


def subsample(idx: np.ndarray, y: np.ndarray, seed: int = 42) -> np.ndarray:
    """All positives plus NEG_PER_POS negatives per positive (train only)."""
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    rng = np.random.default_rng(seed)
    n_neg = min(len(neg), NEG_PER_POS * len(pos))
    keep = np.concatenate([pos, rng.choice(neg, n_neg, replace=False)])
    return np.sort(keep)


def fit_boosting(
    x: np.ndarray, y: np.ndarray, num_round: int, max_depth: int
) -> BoostingMachineClassifier:
    # Snap ML defaults to no L2 and a 0.5 base score. On 1:200 data that diverges or blows up
    # the leaf values; XGBoost's lambda=1 and the class prior keep it stable.
    model = BoostingMachineClassifier(
        num_round=num_round,
        max_depth=max_depth,
        learning_rate=0.1,
        lambda_l2=1.0,
        base_score=float(y.mean()),
        n_jobs=3,
        random_state=42,
    )
    return model.fit(x, y)


def budget_thresholds(p_val: np.ndarray) -> tuple[float, float]:
    """Scores at which PAUSE_BUDGET / HOLD_BUDGET of validation payments would be stopped."""
    return (
        float(np.quantile(p_val, 1 - PAUSE_BUDGET)),
        float(np.quantile(p_val, 1 - HOLD_BUDGET)),
    )


def train() -> None:
    x, y, ts = load_features()
    tr, va, _ = split_index(ts)
    if len(va) == 0:
        raise ValueError("empty validation split; check the ts units")
    tr_s = subsample(tr, y)
    print(f"train {len(tr):,} rows -> {len(tr_s):,} after subsampling, val {len(va):,}")

    grid = []
    best = None
    for num_round, max_depth in GRID:
        t0 = time.time()
        model = fit_boosting(x[tr_s], y[tr_s], num_round, max_depth)
        p_val = model.predict_proba(x[va])[:, 1]
        ap = float(average_precision_score(y[va], p_val))
        secs = time.time() - t0
        grid.append(
            {"num_round": num_round, "max_depth": max_depth, "val_pr_auc": ap, "secs": secs}
        )
        print(f"  rounds={num_round} depth={max_depth} val PR-AUC={ap:.4f} ({secs:.0f}s)")
        if best is None or ap > best[0]:
            best = (ap, num_round, max_depth, model, p_val)

    ap, num_round, max_depth, model, _ = best
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.export_model(str(ARTIFACTS / "guard_model.pmml"), "pmml")
    # Thresholds come from the re-imported model, since that's what the scorer runs.
    served = load_model(ARTIFACTS / "guard_model.pmml")
    p_val = served.predict_proba(x[va])[:, 1]
    t_pause, t_hold = budget_thresholds(p_val)
    (ARTIFACTS / "feature_list.json").write_text(json.dumps(FEATURES, indent=2))
    write_fx_rates(ARTIFACTS / "fx_rates.json")
    (ARTIFACTS / "thresholds.json").write_text(
        json.dumps(
            {
                "t_pause": t_pause,
                "t_hold": t_hold,
                "pause_budget": PAUSE_BUDGET,
                "hold_budget": HOLD_BUDGET,
                "val_pause_rate": float((p_val >= t_pause).mean()),
                "val_hold_rate": float((p_val >= t_hold).mean()),
            },
            indent=2,
        )
    )
    (ARTIFACTS / "train_report.json").write_text(
        json.dumps(
            {
                "num_round": num_round,
                "max_depth": max_depth,
                "learning_rate": 0.1,
                "lambda_l2": 1.0,
                "neg_per_pos": NEG_PER_POS,
                "train_rows": len(tr),
                "train_rows_subsampled": len(tr_s),
                "val_pr_auc": ap,
                "val_pr_auc_served": float(average_precision_score(y[va], p_val)),
                "tree_format": served.booster_.optimized_tree_format_,
                "grid": grid,
            },
            indent=2,
        )
    )
    print(f"picked rounds={num_round} depth={max_depth}; t_pause={t_pause:.4f} t_hold={t_hold:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="feature pass on the first N rows only")
    ap.add_argument("--rebuild", action="store_true", help="recompute cached features")
    args = ap.parse_args()
    if args.limit:
        build_features(args.limit)
        return
    if args.rebuild or not (FEATURE_DIR / "aml_features.npy").exists():
        build_features()
    train()


if __name__ == "__main__":
    sys.exit(main())
