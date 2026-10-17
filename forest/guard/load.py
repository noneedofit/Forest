"""Load the IBM AML (HI-Small) and PaySim CSVs into tidy DataFrames."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

AML_COLUMNS = [
    "ts",
    "from_bank",
    "src",
    "to_bank",
    "dst",
    "amt_recv",
    "cur_recv",
    "amt_paid",
    "cur_paid",
    "fmt",
    "y",
]

# USD per unit. Read off the dataset itself: rows where one side is USD give a fixed implied
# rate per currency (median of amt_paid / amt_recv), so these match how the data was generated.
FX_TO_USD = {
    "US Dollar": 1.0,
    "Euro": 1.1718,
    "Swiss Franc": 1.0929,
    "UK Pound": 1.2918,
    "Yuan": 0.1493,
    "Shekel": 0.2961,
    "Rupee": 0.0136,
    "Yen": 0.0095,
    "Ruble": 0.0129,
    "Canadian Dollar": 0.758,
    "Australian Dollar": 0.7078,
    "Mexican Peso": 0.0473,
    "Saudi Riyal": 0.2666,
    "Brazil Real": 0.1771,
    "Bitcoin": 11900.0,
}

PAYSIM_TYPES = ["TRANSFER", "CASH_OUT"]


def load_aml(path: Path, nrows: int | None = None) -> pd.DataFrame:
    """Read HI-Small_Trans.csv sorted by time, with `amount_usd` and epoch-second `ts`."""
    df = pd.read_csv(
        path,
        header=0,
        names=AML_COLUMNS,
        nrows=nrows,
        dtype={
            "ts": str,
            "from_bank": "category",
            "src": str,
            "to_bank": "category",
            "dst": str,
            "amt_recv": np.float64,
            "cur_recv": "category",
            "amt_paid": np.float64,
            "cur_paid": "category",
            "fmt": "category",
            "y": np.int8,
        },
    )
    missing = set(df["cur_paid"].cat.categories) - set(FX_TO_USD)
    if missing:
        raise ValueError(f"no FX rate for {sorted(missing)}")
    # pandas 3 infers the datetime unit, so pin it to seconds before going to int.
    df["ts"] = (
        pd.to_datetime(df["ts"], format="%Y/%m/%d %H:%M").astype("datetime64[s]").astype("int64")
    )
    df["amount_usd"] = (df["amt_paid"] * df["cur_paid"].map(FX_TO_USD).astype(np.float64)).astype(
        np.float32
    )
    df = df.drop(columns=["amt_recv", "amt_paid"])
    return df.sort_values("ts", kind="stable").reset_index(drop=True)


def load_paysim(path: Path) -> pd.DataFrame:
    """Read PaySim, keeping only TRANSFER and CASH_OUT (the only types with fraud)."""
    df = pd.read_csv(
        path,
        usecols=[
            "step",
            "type",
            "amount",
            "nameOrig",
            "oldbalanceOrg",
            "newbalanceOrig",
            "nameDest",
            "oldbalanceDest",
            "newbalanceDest",
            "isFraud",
        ],
        dtype={"type": "category", "isFraud": np.int8},
    )
    df = df[df["type"].isin(PAYSIM_TYPES)].reset_index(drop=True)
    df["type"] = df["type"].cat.remove_unused_categories()
    return df


def write_fx_rates(path: Path) -> None:
    path.write_text(json.dumps(FX_TO_USD, indent=2))
