"""Шаг 5: малые проверки правил агрегации на синтетических примерах."""
import pandas as pd
import pytest

from prep.merge import aggregate_support

TS = pd.Timestamp


def keys(*pairs):
    return pd.DataFrame([{"customer_id": c, "billing_month": TS(m),
                          "available_at": TS(m) + pd.offsets.MonthBegin(1)} for c, m in pairs])


def ticket(tid, cid, created, available, topic="network", resolved=None, res_av=None, score=None, **flags):
    return {"ticket_id": tid, "customer_id": cid, "created_at": TS(created), "available_at": TS(available),
            "topic": topic, "resolved_at": TS(resolved) if resolved else pd.NaT,
            "resolution_available_at": TS(res_av) if res_av else pd.NaT,
            "satisfaction_score": score,
            "billing_month": TS(created).to_period("M").to_timestamp(),
            "created_tz_assumed": flags.get("tz", 0), "score_invalid": 0, "resolution_invalid": 0}


def frame(rows):
    df = pd.DataFrame(rows)
    df["satisfaction_score"] = df["satisfaction_score"].astype("Int64")
    return df


def row(out, cid, m):
    return out.set_index(["customer_id", "billing_month"]).loc[(cid, TS(m))]


def test_duplicate_row_does_not_double_count():
    t = ticket("T1", "C1", "2026-08-10 10:00", "2026-08-10 20:00")
    out = aggregate_support(frame([t, t]), keys(("C1", "2026-08-01")))
    assert row(out, "C1", "2026-08-01")["support_count"] == 1


def test_two_tickets_same_day_are_both_counted():
    rows = [ticket("T1", "C1", "2026-08-10 10:00", "2026-08-10 20:00"),
            ticket("T2", "C1", "2026-08-10 11:00", "2026-08-10 21:00", topic="billing")]
    r = row(aggregate_support(frame(rows), keys(("C1", "2026-08-01"))), "C1", "2026-08-01")
    assert (r["support_count"], r["support_network_count"], r["support_billing_count"]) == (2, 1, 1)


def test_late_arrival_excluded_boundary_example():
    # Создано 31 августа 23:00, загружено 1 сентября 09:00 -> на 1 сентября 00:00 неизвестно.
    rows = [ticket("T1", "C1", "2026-08-31 23:00", "2026-09-01 09:00")]
    out = aggregate_support(frame(rows), keys(("C1", "2026-08-01"), ("C1", "2026-09-01")))
    assert row(out, "C1", "2026-08-01")["support_count"] == 0
    # и не переносится в сентябрь: месяц создания — август
    assert row(out, "C1", "2026-09-01")["support_count"] == 0


def test_available_exactly_at_snapshot_is_included():
    rows = [ticket("T1", "C1", "2026-08-31 20:00", "2026-09-01 00:00")]
    assert row(aggregate_support(frame(rows), keys(("C1", "2026-08-01"))), "C1", "2026-08-01")["support_count"] == 1


def test_customer_without_tickets_kept_after_left_join():
    rows = [ticket("T1", "C1", "2026-08-10 10:00", "2026-08-10 20:00")]
    out = aggregate_support(frame(rows), keys(("C1", "2026-08-01"), ("C2", "2026-08-01")))
    assert len(out) == 2
    r = row(out, "C2", "2026-08-01")
    assert r["support_count"] == 0 and pd.isna(r["support_score_mean"]) and r["support_score_count"] == 0


def test_resolution_known_only_after_resolution_available_at():
    rows = [ticket("T1", "C1", "2026-08-20 10:00", "2026-08-20 20:00",
                   resolved="2026-08-31 22:00", res_av="2026-09-01 08:00", score=5),
            ticket("T2", "C1", "2026-08-05 10:00", "2026-08-05 20:00",
                   resolved="2026-08-07 10:00", res_av="2026-08-07 18:00", score=3)]
    r = row(aggregate_support(frame(rows), keys(("C1", "2026-08-01"))), "C1", "2026-08-01")
    assert r["support_count"] == 2
    assert r["support_resolved_count"] == 1
    assert r["support_score_count"] == 1 and r["support_score_mean"] == pytest.approx(3.0)


def test_unknown_topic_counted_separately():
    rows = [ticket("T1", "C1", "2026-08-10 10:00", "2026-08-10 20:00", topic="unknown")]
    r = row(aggregate_support(frame(rows), keys(("C1", "2026-08-01"))), "C1", "2026-08-01")
    assert (r["support_count"], r["support_unknown_topic_count"], r["support_network_count"]) == (1, 1, 0)
