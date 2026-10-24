"""Проверки результата полного запуска и воспроизводимости."""

import json

import pandas as pd
import pytest
from prep.common import parse_current
from prep.load import CSV_KW
from prep.pipeline import run


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    a = tmp_path_factory.mktemp("run_a")
    b = tmp_path_factory.mktemp("run_b")
    return run(output_dir=a, verbose=False), run(output_dir=b, verbose=False), a


def read(path):
    return pd.read_csv(path, **CSV_KW)


def test_reproducible_outputs(runs):
    s1, s2, _ = runs
    assert s1["output_sha256"] == s2["output_sha256"]
    assert s1["rows_out"] == s2["rows_out"]


def test_raw_files_unchanged(runs):
    assert runs[0]["raw_checksums_ok"]


def test_sqlite_and_csv_backup_are_same_table(runs):
    assert runs[0]["sqlite_equals_csv_backup"]


def test_customer_month_does_not_multiply_billing(runs):
    _, _, out = runs
    billing, cm = (
        read(out / "billing_clean.csv"),
        read(out / "customer_month.csv"),
    )
    assert len(cm) == len(billing)
    assert not cm.duplicated(["customer_id", "billing_month"]).any()
    for col in ["charged_rub", "paid_rub", "closing_debt_rub"]:
        assert pd.to_numeric(cm[col]).sum() == pytest.approx(
            pd.to_numeric(billing[col]).sum()
        )


def test_no_subscriber_removed(runs):
    _, _, out = runs
    subs = read(out / "subscribers_clean.csv")
    assert len(subs) == 2600 and subs["customer_id"].is_unique
    assert (subs["activation_date"] != "").all()


def test_support_keys_and_links(runs):
    _, _, out = runs
    sup, subs = (
        read(out / "support_clean.csv"),
        read(out / "subscribers_clean.csv"),
    )
    assert sup["ticket_id"].is_unique
    assert sup["customer_id"].isin(set(subs["customer_id"])).all()
    q = read(out / "quarantine_support.csv")
    assert (q["reason"] != "").all()
    assert not set(q["ticket_id"]) & set(sup["ticket_id"])


def test_all_support_accounted(runs):
    s = runs[0]
    inv = {i["source"]: i["rows"] for i in s["inventory"]}
    raw = inv["support_legacy (CSV)"] + inv["support_current (SQLite)"]
    copies = 30 + 100
    assert (
        s["rows_out"]["support_clean"] + s["rows_out"]["quarantine_support"]
        == raw - copies
    )


def test_availability_checks_pass(runs):
    assert runs[0]["availability_checks_passed"]


def test_no_errors_left_after_cleaning(runs):
    assert runs[0]["error_cells_or_rows_after"] == 0


def test_timezone_rules():
    s = pd.Series(["2026-08-31T21:00:00Z", "2026-09-01T00:00:00", ""])
    msk, naive = parse_current(s)
    assert msk.iloc[0] == pd.Timestamp("2026-09-01 00:00:00")
    assert msk.iloc[1] == pd.Timestamp("2026-09-01 00:00:00")
    assert pd.isna(msk.iloc[2]) and naive.tolist() == [False, True, False]
