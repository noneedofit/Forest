"""Guard metrics on the time split, an LR baseline and a random-split contrast.

python -m eval.eval_guard   ->  eval/results/guard.json
"""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from forest.guard.load import load_aml
from forest.guard.score import Guard, decide, load_model
from forest.guard.train import (
    ARTIFACTS,
    RAW,
    TEST_END,
    budget_thresholds,
    fit_boosting,
    load_features,
    split_index,
    subsample,
)

OUT = Path(__file__).resolve().parent / "results" / "guard.json"


def at_threshold(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    flagged = p >= t
    tp = int((flagged & (y == 1)).sum())
    fp = int((flagged & (y == 0)).sum())
    fn = int((~flagged & (y == 1)).sum())
    return {
        "threshold": float(t),
        "flagged_rate": float(flagged.mean()),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": int(len(y) - tp - fp - fn),
    }


def metrics(y: np.ndarray, p: np.ndarray, t_pause: float, t_hold: float) -> dict:
    return {
        "rows": len(y),
        "positives": int(y.sum()),
        "pr_auc": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "pause_1pct": at_threshold(y, p, t_pause),
        "hold_0_1pct": at_threshold(y, p, t_hold),
    }


def fit_lr(x: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(x, y)


def positives_per_day(y: np.ndarray, ts: np.ndarray) -> dict[str, int]:
    days = ts // 86400
    out = {}
    for d in np.unique(days):
        day = datetime.fromtimestamp(int(d) * 86400, tz=UTC).strftime("%Y-%m-%d")
        out[day] = int(y[days == d].sum())
    return out


def run_split(x, y, tr, va, te, num_round: int, max_depth: int) -> dict:
    """Same recipe as train.py on an arbitrary split: subsample, fit, val thresholds, test."""
    tr_s = subsample(tr, y)
    model = fit_boosting(x[tr_s], y[tr_s], num_round, max_depth)
    p_va = model.predict_proba(x[va])[:, 1]
    t_pause, t_hold = budget_thresholds(p_va)
    return metrics(y[te], model.predict_proba(x[te])[:, 1], t_pause, t_hold)


def replay_check(model, th: dict, x: np.ndarray, start: int, n: int = 20_000) -> dict:
    """Push the first `n` test payments through Guard.score from the saved end-of-val state.
    Its decisions should match the offline batch scores row for row."""
    df = load_aml(RAW).iloc[start : start + n]
    guard = Guard(ARTIFACTS)
    lat, live = [], []
    for r in df.itertuples(index=False):
        d = guard.score(
            {
                "ts": r.ts,
                "src": r.src,
                "dst": r.dst,
                "from_bank": r.from_bank,
                "to_bank": r.to_bank,
                "amount": r.amt_paid,
                "currency": r.cur_paid,
                "currency_recv": r.cur_recv,
                "fmt": r.fmt,
            }
        )
        lat.append(d.latency_ms)
        live.append(d.decision)
    p = model.predict_proba(x[start : start + n])[:, 1]
    offline = [decide(v, th["t_pause"], th["t_hold"], False) for v in p]
    lat = np.array(lat)
    return {
        "rows": len(live),
        "decision_agreement": float(np.mean([a == b for a, b in zip(live, offline)])),
        "decisions": {k: live.count(k) for k in ("allow", "pause", "hold")},
        "tree_format": guard.tree_format,
        "latency_ms": {
            "p50": float(np.percentile(lat, 50)),
            "p99": float(np.percentile(lat, 99)),
            "mean": float(lat.mean()),
        },
        "note": "one call = feature update + Snap ML predict + policy, single thread",
    }


def row(name: str, m: dict) -> str:
    pa, ho = m["pause_1pct"], m["hold_0_1pct"]
    return (
        f"| {name} | {m['pr_auc']:.3f} | {m['roc_auc']:.3f} | {pa['recall']:.3f} | "
        f"{pa['precision']:.3f} | {ho['recall']:.3f} | {ho['precision']:.3f} |"
    )


def main() -> None:
    x, y, ts = load_features()
    tr, va, te = split_index(ts)
    report = json.loads((ARTIFACTS / "train_report.json").read_text())
    th = json.loads((ARTIFACTS / "thresholds.json").read_text())

    model = load_model(ARTIFACTS / "guard_model.pmml")
    snap = {
        "val": metrics(y[va], model.predict_proba(x[va])[:, 1], th["t_pause"], th["t_hold"]),
        "test": metrics(y[te], model.predict_proba(x[te])[:, 1], th["t_pause"], th["t_hold"]),
    }

    tr_s = subsample(tr, y)
    lr = fit_lr(x[tr_s], y[tr_s])
    lr_pause, lr_hold = budget_thresholds(lr.predict_proba(x[va])[:, 1])
    lr_test = metrics(y[te], lr.predict_proba(x[te])[:, 1], lr_pause, lr_hold)

    n = int(np.searchsorted(ts, TEST_END))
    rng = np.random.default_rng(42)
    perm = rng.permutation(n)
    a, b = int(0.6 * n), int(0.8 * n)
    rand = run_split(
        x,
        y,
        np.sort(perm[:a]),
        np.sort(perm[a:b]),
        np.sort(perm[b:]),
        report["num_round"],
        report["max_depth"],
    )

    replay = replay_check(model, th, x, int(te[0]))
    print(f"replay: {replay['decision_agreement']:.4%} agree, latency {replay['latency_ms']}")

    result = {
        "data": "IBM AML HI-Small (5,078,345 transfers)",
        "split": "time: train Sep 1-6, val Sep 7-8, test Sep 9-10 (2022)",
        "excluded_tail": {
            "note": "Sep 11-18: normal traffic has stopped, rows are mostly laundering",
            "rows": int(len(ts) - n),
            "positives": int(y[n:].sum()),
        },
        "budgets": {"pause": th["pause_budget"], "hold": th["hold_budget"]},
        "thresholds_from": "validation period only",
        "model": {
            k: report[k]
            for k in ("num_round", "max_depth", "learning_rate", "lambda_l2", "neg_per_pos")
        },
        "tree_format": report["tree_format"],
        "grid": report["grid"],
        "positives_per_day": positives_per_day(y, ts),
        "snapml_boosting": snap,
        "logreg_baseline": {"test": lr_test},
        "random_split_leaky": {
            "note": "same recipe on a shuffled 60/20/20 split of Sep 1-10; mixes future into train",
            "test": rand,
        },
        "live_replay": replay,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))

    print("| model | PR-AUC | ROC-AUC | recall@1% | prec@1% | recall@0.1% | prec@0.1% |")
    print("|---|---|---|---|---|---|---|")
    print(row("Snap ML boosting, test (time split)", snap["test"]))
    print(row("Snap ML boosting, val", snap["val"]))
    print(row("Logistic regression, test", lr_test))
    print(row("Snap ML, random split (leaky)", rand))


if __name__ == "__main__":
    main()
