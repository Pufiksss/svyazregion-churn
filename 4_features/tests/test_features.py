"""Проверки этапа 4 на синтетических примерах и полном запуске."""
import copy

import numpy as np
import pandas as pd
import pytest

from feat import build as B
from feat.pipeline import run

T = pd.Timestamp("2026-07-01")
CFG = {"window_months": 3, "active_mode": "before_events", "horizon": "calendar_month",
       "observation_end_exclusive": "2026-10-01", "digital_last_report_max_age_months": 3}


def toy():
    subs = pd.DataFrame({"customer_id": ["A", "N"], "activation_date": pd.to_datetime(["2024-01-10", "2026-06-15"]),
                         "region": "R", "customer_segment": "individual", "acquisition_channel": "online"})
    ev = pd.DataFrame({"customer_id": ["A", "N"], "event_type": "activation", "event_date": subs["activation_date"]})
    months = pd.to_datetime(["2026-04-01", "2026-05-01", "2026-06-01"])
    bill = pd.DataFrame({"customer_id": ["A"] * 3 + ["N"], "billing_month": list(months) + [months[2]],
                         "tariff": "Basic", "charged_rub": [450, 450, 450, 240], "paid_rub": [450, 0, 900, 0],
                         "closing_debt_rub": [0, 450, 0, 240], "traffic_gb": [10, np.nan, 20, 5],
                         "voice_minutes": 100, "active_days": [30, 31, 30, 16], "payment_delay_days": 0,
                         "tariff_changes_count": 0})
    bill["available_at"] = bill["billing_month"] + B.MS
    sup = pd.DataFrame({"ticket_id": ["t1", "t2"], "customer_id": ["A", "A"],
                        "created_at": pd.to_datetime(["2026-06-30 23:00", "2026-06-10 10:00"]),
                        "available_at": pd.to_datetime(["2026-07-01 09:00", "2026-06-10 20:00"]),
                        "topic": ["billing", "network"], "resolved_at": pd.NaT, "resolution_available_at": pd.NaT})
    dig = pd.DataFrame({"customer_id": ["A", "N"], "usage_month": pd.to_datetime(["2026-06-01", "2026-06-01"]),
                        "available_at": pd.to_datetime(["2026-07-01 00:00:00", "2026-07-01 00:00:01"]),
                        "observation_status": ["observed", "observed"], "observed_days": [30, 16],
                        "app_sessions": [0, 5], "payment_attempts": [0, 1], "payment_failures": [0, 1],
                        "selfservice_actions": [0, 1]})
    return {"subscribers": subs, "events": ev, "billing": bill, "support": sup, "digital": dig}


def snap(data):
    return B.build_snapshot(T, data, CFG)[0].set_index("customer_id")


def test_short_history_not_zero_filled():
    f = snap(toy())
    assert f.loc["N", "n_billing_months_3m"] == 1 and f.loc["N", "debt_months_3m"] == 1
    assert np.isnan(f.loc["N", "debt_change_1m_rub"])


def test_missing_month_gives_unknown_difference():
    d = toy(); d["billing"] = d["billing"][d["billing"]["billing_month"] != pd.Timestamp("2026-05-01")]
    assert np.isnan(snap(d).loc["A", "debt_change_1m_rub"])


def test_traffic_per_day_uses_known_pairs_only():
    f = snap(toy())
    assert f.loc["A", "traffic_per_active_day_3m"] == pytest.approx(30 / 60)


def test_pay_ratio_above_one_allowed():
    assert snap(toy()).loc["A", "pay_to_charge_ratio_3m"] == pytest.approx(1350 / 1350)


def test_digital_boundary_and_zero_attempts():
    f = snap(toy())
    assert f.loc["A", "digital_status_prev"] == "observed"
    assert f.loc["A", "app_sessions_prev"] == 0
    assert np.isnan(f.loc["A", "payment_fail_rate_prev"])
    assert f.loc["N", "digital_status_prev"] == "not_published_at_T"
    assert np.isnan(f.loc["N", "app_sessions_prev"])


def test_late_support_ticket_excluded_and_future_does_not_change():
    f = snap(toy())
    assert f.loc["A", "support_3m"] == 1 and f.loc["A", "support_network_3m"] == 1
    d = copy.deepcopy(toy())
    d["support"].loc[0, "available_at"] = T
    assert snap(d).loc["A", "support_3m"] == 2


def test_label_change_does_not_change_features():
    d = toy(); f0 = snap(d)
    d2 = copy.deepcopy(d)
    d2["events"] = pd.concat([d2["events"], pd.DataFrame({"customer_id": ["A"], "event_type": ["termination"],
                                                          "event_date": [pd.Timestamp("2026-07-15")]})])
    pd.testing.assert_frame_equal(f0, snap(d2))
    assert B.build_labels(T, d2, CFG).set_index("customer_id").loc["A", "churn_target"] == 1


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    return run(output_dir=tmp_path_factory.mktemp("a"), verbose=False), run(output_dir=tmp_path_factory.mktemp("b"), verbose=False)


def test_full_run_checks_pass(runs):
    assert runs[0]["checks"]["passed"].all(), runs[0]["checks"][~runs[0]["checks"]["passed"]]


def test_reproducible(runs):
    assert runs[0]["summary"]["csv_sha256"] == runs[1]["summary"]["csv_sha256"]
