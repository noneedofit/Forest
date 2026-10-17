import math

import pytest

from forest.guard.features import DAY, FEATURES, WEEK, FeatureState, to_epoch
from forest.guard.load import load_aml

T0 = to_epoch("2022/09/01 00:00")


def pay(state: FeatureState, t: int, src: str, dst: str, amt: float, **kw) -> dict:
    txn = {"ts": t, "src": src, "dst": dst, "amount_usd": amt, **kw}
    return dict(zip(FEATURES, state.update_and_featurise(txn)))


def test_toy_stream():
    st = FeatureState()
    f1 = pay(st, T0, "A", "X", 100)
    assert f1["s_n_prior"] == 0
    assert f1["s_new_payee"] == 1
    assert f1["s_secs_since_last"] == WEEK
    assert f1["r_age_hours"] == 0
    assert f1["r_n_senders_24h"] == 0

    f2 = pay(st, T0 + 60, "A", "X", 100)
    assert f2["s_new_payee"] == 0
    assert f2["s_secs_since_last"] == 60
    assert f2["r_n_senders_24h"] == 1

    f3 = pay(st, T0 + 120, "A", "Y", 100)
    assert f3["s_new_payee"] == 1
    assert f3["s_n_payees"] == 1
    assert f3["amt_z"] == 0  # fewer than 3 prior payments

    pay(st, T0 + 180, "B", "X", 50)

    f5 = pay(st, T0 + 240, "A", "X", 10_000)
    assert f5["s_n_prior"] == 3
    assert f5["s_n_payees"] == 2
    assert f5["s_n_last_1h"] == 3
    assert f5["s_std_log_amt"] == 0
    assert f5["amt_z"] == pytest.approx((math.log1p(10_000) - math.log1p(100)) / 0.25, rel=1e-6)
    assert f5["r_n_senders_24h"] == 2
    assert f5["r_n_new_senders_24h"] == 2
    assert f5["r_in_log_24h"] == pytest.approx(math.log1p(250))

    # A's first two payments to X fall out of the 24 h window; B's and A's big one stay.
    f6 = pay(st, T0 + DAY + 120, "C", "X", 100)
    assert f6["r_n_senders_24h"] == 2
    assert f6["r_n_new_senders_24h"] == 1
    assert f6["r_in_log_24h"] == pytest.approx(math.log1p(10_050))
    assert f6["r_age_hours"] == pytest.approx((DAY + 120) / 3600)
    assert f6["s_new_payee"] == 1


def test_fan_in_after_three_senders():
    st = FeatureState()
    for i, src in enumerate("ABC"):
        pay(st, T0 + i * 60, src, "M", 500)
    f = pay(st, T0 + 300, "D", "M", 500)
    assert f["r_n_senders_24h"] == 3
    assert f["r_n_new_senders_24h"] == 3
    f = pay(st, T0 + 360, "A", "M", 500)
    assert f["r_n_senders_24h"] == 4
    assert f["r_n_new_senders_24h"] == 4


def test_sender_windows_expire():
    st = FeatureState()
    pay(st, T0, "A", "X", 100)
    pay(st, T0 + 2 * 3600, "A", "X", 100)
    f = pay(st, T0 + DAY + 60, "A", "X", 100)
    assert f["s_n_last_24h"] == 1
    assert f["s_n_last_1h"] == 0
    assert f["s_sum_last_24h_log"] == pytest.approx(math.log1p(100))


def test_out_in_ratio_for_forwarding_receiver():
    st = FeatureState()
    pay(st, T0, "A", "M", 100)
    pay(st, T0 + 60, "M", "Z", 90)
    f = pay(st, T0 + 120, "B", "M", 10)
    assert f["r_out_in_ratio_24h"] == pytest.approx(0.9)


def test_transaction_fields():
    st = FeatureState()
    f = pay(
        st,
        to_epoch("2022/09/05 13:30"),
        "A",
        "A",
        1,
        fmt="ACH",
        from_bank="001",
        to_bank="002",
        currency="Euro",
        currency_recv="US Dollar",
    )
    assert f["hour"] == 13
    assert f["dow"] == 0  # a Monday
    assert f["is_self"] == 1
    assert f["cross_bank"] == 1
    assert f["cur_mismatch"] == 1
    assert f["fmt_ach"] == 1
    assert sum(v for k, v in f.items() if k.startswith("fmt_")) == 1


def test_unknown_format_is_all_zero():
    f = pay(FeatureState(), T0, "A", "B", 10, fmt="UPI")
    assert sum(v for k, v in f.items() if k.startswith("fmt_")) == 0


def test_to_epoch_forms_agree():
    from datetime import datetime

    assert to_epoch("2022-09-01T00:00:00") == T0
    assert to_epoch(datetime(2022, 9, 1)) == T0
    assert to_epoch(float(T0)) == T0


def test_load_aml_epoch_seconds_and_usd(tmp_path):
    csv = tmp_path / "aml.csv"
    csv.write_text(
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
        "Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:20,010,AAA,010,BBB,10.0,Euro,10.0,Euro,ACH,1\n"
        "2022/09/01 00:05,00795,CCC,001,AAA,5.0,US Dollar,5.0,US Dollar,Cheque,0\n"
    )
    df = load_aml(csv)
    assert df["ts"].tolist() == [T0 + 5 * 60, T0 + 20 * 60]
    assert df["src"].tolist() == ["CCC", "AAA"]
    assert df["from_bank"].astype(str).tolist() == ["00795", "010"]
    assert df["amount_usd"].tolist() == pytest.approx([5.0, 11.718])
    assert df["y"].tolist() == [0, 1]


def test_restore_undoes_a_stopped_payment():
    st = FeatureState()
    pay(st, T0, "A", "X", 100)
    pay(st, T0 + 60, "B", "X", 100)
    s, d = st.account_id("A"), st.account_id("M")
    saved = st.save(s, d)
    pay(st, T0 + 120, "A", "M", 900)
    st.restore(saved)

    fresh = FeatureState()
    pay(fresh, T0, "A", "X", 100)
    pay(fresh, T0 + 60, "B", "X", 100)
    a = pay(st, T0 + 180, "A", "M", 50)
    b = pay(fresh, T0 + 180, "A", "M", 50)
    assert a == b
    assert a["s_new_payee"] == 1
    assert a["r_age_hours"] == 0
