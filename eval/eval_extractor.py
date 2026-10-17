"""Extractor precision/recall on planted entities, scam-type classifier reports, stage rules and
per-message latency.

python -m eval.eval_extractor   ->  eval/results/extractor.json
"""

import csv
import json
import random
import re
import string
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
from sklearn.model_selection import StratifiedGroupKFold

from forest.honeypot.scenario import load_all, render
from forest.intel import classify
from forest.intel.extract import KNOWN_PSP, extract
from forest.intel.stage import stage

OUT = Path(__file__).resolve().parent / "results" / "extractor.json"
SEED = 42
N_MESSAGES = 300
KINDS = ("upi", "account", "phone", "url", "amount")

NAMES = ["ramesh", "sunita", "vikram", "anil", "meena", "rohit", "priya", "kavita", "sanjay"]
TAILS = ["verify", "pay", "help", "desk", "claim", "fee", "kyc", "care", "refund"]
UNKNOWN_PSP = ["newbank", "fastpay", "quickupi", "zippay"]
BANK_CODES = ["FRST", "TEST", "DEMO", "NOVA"]
HINDI_NUMS = {"ek": 1, "do": 2, "teen": 3, "paanch": 5, "das": 10, "bees": 20, "pachaas": 50}


def canon_amount(x: str | float) -> str:
    return f"{float(x):g}"


def plant_upi(rng: random.Random) -> tuple[str, dict]:
    sep = rng.choice([".", "_", ""])
    tail = rng.choice(TAILS) if rng.random() < 0.7 else ""
    local = rng.choice(NAMES) + (sep + tail if tail else "") + str(rng.randint(10, 9999))
    known = rng.random() < 0.8
    psp = rng.choice(sorted(KNOWN_PSP)) if known else rng.choice(UNKNOWN_PSP)
    truth = {"kind": "upi", "norm": f"{local}@{psp}", "form": "plain"}
    if rng.random() < 0.65:
        return truth["norm"], truth
    spoken_local = local.replace(".", " dot ")
    options = ["at the rate", "at the rate of", "@"] + (["at"] if known else [])
    truth["form"] = "spoken"
    return f"{spoken_local} {rng.choice(options)} {psp}", truth


def plant_account(rng: random.Random) -> tuple[str, list[dict]]:
    n = rng.randint(9, 18)
    digits = str(rng.randint(1, 9)) + "".join(rng.choice(string.digits) for _ in range(n - 1))
    style = rng.choice(["plain", "spaced", "dashed"])
    sep = {"plain": "", "spaced": " ", "dashed": "-"}[style]
    shown = sep.join(digits[i : i + 4] for i in range(0, n, 4)) if sep else digits
    key = rng.choice(["Account number", "a/c", "A/C no", "khata number", "account no."])
    text = f"{key} {shown}"
    truth = [{"kind": "account", "norm": digits, "form": style}]
    if rng.random() < 0.7:
        code = rng.choice(BANK_CODES) + "0" + "".join(rng.choices(string.digits, k=6))
        text += f", IFSC {code}"
        truth[0]["ifsc"] = code
    return text, truth


def plant_phone(rng: random.Random) -> tuple[str, dict]:
    ten = rng.choice("6789") + "".join(rng.choices(string.digits, k=9))
    shown = rng.choice(
        [ten, f"+91 {ten[:5]} {ten[5:]}", f"+91-{ten}", f"{ten[:5]}-{ten[5:]}", f"+91{ten}"]
    )
    return shown, {"kind": "phone", "norm": "+91" + ten, "form": "plain"}


def plant_amount(rng: random.Random) -> tuple[str, dict]:
    form = rng.choice(["rs", "lakh", "k", "rupees", "hindi"])
    if form == "rs":
        n = rng.randint(500, 99999)
        text, value = f"Rs {n:,}", n
    elif form == "lakh":
        x = rng.choice([1, 1.5, 2, 2.5, 3, 5])
        text, value = f"₹{x} lakh", x * 100_000
    elif form == "k":
        k = rng.randint(2, 50)
        text, value = f"INR {k}k", k * 1000
    elif form == "rupees":
        n = rng.randint(100, 9999)
        text, value = f"{n} rupees", n
    else:
        word = rng.choice(list(HINDI_NUMS))
        unit, mult = rng.choice([("hazaar", 1000), ("lakh", 100_000)])
        text, value = f"{word} {unit} rupees", HINDI_NUMS[word] * mult
    return text, {"kind": "amount", "norm": canon_amount(value), "form": form}


def plant_url(rng: random.Random) -> tuple[str, dict]:
    host = f"{rng.choice(NAMES)}-{rng.choice(TAILS)}"
    if rng.random() < 0.5:
        url = f"http://{host}.{rng.choice(['xyz', 'top', 'in', 'com'])}/app/{rng.choice(TAILS)}.apk"
        form = "apk"
    else:
        url = f"{host}.in/pay"
        form = "bare"
    return url, {"kind": "url", "norm": url.lower(), "form": form}


SENTENCES = {
    "upi": ["Transfer it to {}", "Pay on {} abhi", "UPI id hai {}", "Send the amount to {}"],
    "account": ["{}", "Bank transfer: {}"],
    "phone": ["Call me on {}", "WhatsApp {}", "Helpline {}"],
    "amount": ["Fee is {}", "{} abhi bhejo", "Amount due {}"],
    "url": ["Open {} now", "Install from {}.", "Details at {}"],
}


def decoy(rng: random.Random) -> str:
    def d(n: int) -> str:
        return "".join(rng.choices(string.digits, k=n))

    return rng.choice(
        [
            f"Aadhaar {d(4)} {d(4)} {d(4)} is linked",
            f"The OTP is {d(6)}",
            f"Case number {d(5)}",
            f"Write to help{d(3)}@gmail.com",
            f"Reference {d(8)}",
            "Call before 5 baje",
        ]
    )


def bases(rng: random.Random) -> list[str]:
    """Digit-free scammer lines, so every number in a test message is a planted one."""
    scenario = [
        line
        for sc in load_all().values()
        for step in sc["steps"]
        if "reveal" not in step
        for line in (render(x, None) for x in step["lines"])
        if not re.search(r"\d", line)
    ]
    path = classify.RAW / "scam_calls_hinglish.csv"
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r["text"] for r in csv.DictReader(f) if r["label"] == "1"]
    calls = sorted({t for t in rows if not re.search(r"\d", t)})
    return [rng.choice(scenario) for _ in range(N_MESSAGES // 2)] + [
        rng.choice(calls) for _ in range(N_MESSAGES - N_MESSAGES // 2)
    ]


def make_messages() -> list[tuple[str, list[dict]]]:
    rng = random.Random(SEED)
    out = []
    for base in bases(rng):
        kinds = rng.sample(KINDS, rng.randint(1, 3))
        parts, truth = [], []
        for kind in kinds:
            if kind == "account":
                text, t = plant_account(rng)
                parts.append(rng.choice(SENTENCES[kind]).format(text))
                truth += t
                continue
            planter = {
                "upi": plant_upi,
                "phone": plant_phone,
                "amount": plant_amount,
                "url": plant_url,
            }[kind]
            text, t = planter(rng)
            parts.append(rng.choice(SENTENCES[kind]).format(text))
            truth.append(t)
        if rng.random() < 0.4:
            parts.append(decoy(rng))
        out.append((" ".join([base, *parts]), truth))
    return out


def norm_of(e) -> str:
    return canon_amount(e.normalized) if e.kind == "amount" else e.normalized


def score_extractor(messages: list[tuple[str, list[dict]]]) -> dict:
    tp, fp, fn = Counter(), Counter(), Counter()
    by_form: dict[str, Counter] = {}
    ifsc_ok = ifsc_total = 0
    for text, truth in messages:
        found = extract(text)
        pred = Counter((e.kind, norm_of(e)) for e in found)
        for t in truth:
            key = (t["kind"], t["norm"])
            form = by_form.setdefault(f"{t['kind']}:{t['form']}", Counter())
            form["n"] += 1
            if pred[key] > 0:
                pred[key] -= 1
                tp[t["kind"]] += 1
                form["found"] += 1
                if "ifsc" in t:
                    ifsc_total += 1
                    hit = next(
                        e for e in found if e.kind == "account" and e.normalized == t["norm"]
                    )
                    ifsc_ok += hit.meta.get("ifsc") == t["ifsc"]
            else:
                fn[t["kind"]] += 1
        for (kind, _), left in pred.items():
            fp[kind] += left
    per_kind = {}
    for k in KINDS:
        p = tp[k] / (tp[k] + fp[k]) if tp[k] + fp[k] else 0.0
        r = tp[k] / (tp[k] + fn[k]) if tp[k] + fn[k] else 0.0
        per_kind[k] = {
            "support": tp[k] + fn[k],
            "tp": tp[k],
            "fp": fp[k],
            "fn": fn[k],
            "precision": round(p, 4),
            "recall": round(r, 4),
        }
    return {
        "n_messages": len(messages),
        "per_kind": per_kind,
        "recall_by_form": {k: round(v["found"] / v["n"], 4) for k, v in sorted(by_form.items())},
        "ifsc_attached_correctly": {"correct": ifsc_ok, "of": ifsc_total},
    }


def clf_report(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict:
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    cm_labels = sorted(set(y_true) | set(y_pred))
    scam_true = [y != "legit" for y in y_true]
    scam_pred = [y != "legit" for y in y_pred]
    return {
        "n_test": len(y_true),
        "macro_f1": round(float(f1_score(y_true, y_pred, labels=labels, average="macro")), 4),
        "scam_vs_legit_f1": round(float(f1_score(scam_true, scam_pred)), 4),
        "per_class": {
            lab: {
                "precision": round(float(p[i]), 4),
                "recall": round(float(r[i]), 4),
                "f1": round(float(f[i]), 4),
                "support": int(s[i]),
            }
            for i, lab in enumerate(labels)
        },
        "confusion": {
            "labels": cm_labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=cm_labels).tolist(),
        },
    }


def cross_source(train: classify.Rows, test: classify.Rows) -> dict:
    model = classify.fit(train)
    trained = set(model.classes_)
    kept = [r for r in test if r[1] in trained]
    texts, y_true = zip(*kept, strict=True)
    y_pred = list(model.predict(list(texts)))
    labels = sorted(set(y_true))
    out = clf_report(list(y_true), y_pred, labels)
    out["n_train"] = len(train)
    out["test_rows_with_label_unseen_in_training"] = len(test) - len(kept)
    return out


def in_distribution(rows: classify.Rows) -> dict:
    rows, _ = classify.dedupe(rows)
    texts = [t for t, _ in rows]
    y = [lab for _, lab in rows]
    groups = [classify.group_key(t) for t in texts]
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=classify.SPLIT_SEED)
    train_i, test_i = next(splitter.split(texts, y, groups))
    assert not {groups[i] for i in train_i} & {groups[i] for i in test_i}
    model = classify.fit([rows[i] for i in train_i])
    y_true = [y[i] for i in test_i]
    y_pred = list(model.predict([texts[i] for i in test_i]))
    out = clf_report(y_true, y_pred, sorted(set(y_true)))
    out["n_train"] = len(train_i)
    out["n_groups"] = len(set(groups))
    return out


def stage_accuracy() -> dict:
    wrong, total = [], 0
    for sc in load_all().values():
        for step in sc["steps"]:
            item = sc["collect"][step["reveal"]] if "reveal" in step else None
            for line in step["lines"]:
                total += 1
                text = render(line, item)
                got = stage(text)
                if got != step["stage"]:
                    wrong.append({"scenario": sc["id"], "expected": step["stage"], "got": got})
    return {"lines": total, "correct": total - len(wrong), "mismatches": wrong}


def latency(messages: list[tuple[str, list[dict]]]) -> dict:
    texts = [t for t, _ in messages]
    classify.classify(texts[0])
    both, ext_only = [], []
    for t in texts:
        t0 = time.perf_counter()
        extract(t)
        t1 = time.perf_counter()
        classify.classify(t)
        t2 = time.perf_counter()
        ext_only.append((t1 - t0) * 1000)
        both.append((t2 - t0) * 1000)

    def pct(a: list[float]) -> dict:
        return {
            "p50": round(float(np.percentile(a, 50)), 3),
            "p99": round(float(np.percentile(a, 99)), 3),
        }

    return {"n": len(texts), "extract_ms": pct(ext_only), "extract_plus_classify_ms": pct(both)}


def main() -> None:
    sources = classify.load_sources()
    shade, hing, seed = (sources[k][0] for k in ("shade63", "hinglish_calls", "job_task_seed"))
    norm_overlap = {classify.normalise(t) for t, _ in shade} & {
        classify.normalise(t) for t, _ in hing
    }
    messages = make_messages()
    a_to_b = cross_source(shade, hing)
    b_to_a = cross_source(hing, shade)
    headline = round((a_to_b["macro_f1"] + b_to_a["macro_f1"]) / 2, 4)
    extractor = score_extractor(messages)

    classify.train_and_save()
    result = {
        "generated": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "headline": {
            "classifier_cross_source_macro_f1_mean": headline,
            "extractor_per_kind_f1": {
                k: round(
                    2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"])
                    if v["precision"] + v["recall"]
                    else 0.0,
                    4,
                )
                for k, v in extractor["per_kind"].items()
            },
        },
        "extractor_planted": extractor
        | {
            "note": "Messages are scenario lines and Hinglish call openers with fictitious "
            "entities planted from our own templates, so this is a regression-style check "
            "and an upper bound on real-chat performance. 40% carry a decoy (Aadhaar-like "
            "number, OTP, case number, email) that must not be extracted. Known gap by design: "
            "a 10-digit account number starting 6-9 is read as a phone number (all remaining "
            "account misses and phone false positives)."
        },
        "classifier": {
            "labels_without_training_data": ["other", "family_emergency"],
            "notes": [
                (
                    "scam_calls_hinglish.csv reuses the same pool of texts under bank_kyc, "
                    "amazon, aadhaar, lottery and relative, so those categories contradict "
                    "themselves. Conflicting texts are dropped; only police_* and legit "
                    "survive from that file."
                ),
                (
                    "Cross-source is a hard test: English bank SMS against Hinglish call "
                    "openers, and 'legit' means a different thing in each source."
                ),
            ],
            "dedupe": {name: stats for name, (_, stats) in sources.items()}
            | {"normalised_texts_shared_by_shade63_and_hinglish": len(norm_overlap)},
            "cross_source": {
                "shade63_to_hinglish_calls": a_to_b,
                "hinglish_calls_to_shade63": b_to_a,
                "macro_f1_mean": headline,
            },
            "in_distribution": in_distribution(shade + hing + seed),
            "job_task_seed": {
                "rows": len(seed),
                "note": "hand-written from public advisories, weak class",
            },
        },
        "stage_rules_on_scenario_lines": stage_accuracy(),
        "latency": latency(messages),
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["headline"], indent=2))


if __name__ == "__main__":
    main()
