"""Месячная таблица customer_month: одна строка на (customer_id, billing_month).

Правило доступности: строка известна в момент snapshot = available_at биллинга (00:00 МСК 1-го числа
следующего месяца). Обращение входит в агрегат месяца m, если
  created_at ∈ [m, m + 1 мес.) по МСК  и  available_at обращения ≤ snapshot.
Закрытие и оценка учитываются, только если resolution_available_at ≤ snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TOPIC_COUNTS = {"network": "support_network_count", "billing": "support_billing_count",
                "tariff": "support_tariff_count", "services": "support_services_count",
                "unknown": "support_unknown_topic_count"}
SUPPORT_COLS = ["support_count", *TOPIC_COUNTS.values(), "support_resolved_count",
                "support_score_count", "support_score_mean", "support_flagged_count",
                "support_quarantined_linked_count"]


def aggregate_support(support: pd.DataFrame, keys: pd.DataFrame,
                      quarantine: pd.DataFrame | None = None) -> pd.DataFrame:
    """support — типизированные чистые обращения (даты datetime, billing_month — 1-е число).
    keys — (customer_id, billing_month, available_at) строк биллинга.
    Возвращает агрегаты на ключ (customer_id, billing_month) только для пар из keys.
    """
    s = support.drop_duplicates()  # защита: полная копия не удваивает счётчик
    s = s.merge(keys, on=["customer_id", "billing_month"], how="inner", suffixes=("", "_snapshot"))
    s = s[s["available_at"] <= s["available_at_snapshot"]]
    res_ok = s["resolution_available_at"].notna() & (s["resolution_available_at"] <= s["available_at_snapshot"])
    s = s.assign(
        is_resolved=res_ok.astype(int),
        score_avail=s["satisfaction_score"].astype("Float64").where(res_ok),
        flagged=((s["created_tz_assumed"] + s["score_invalid"] + s["resolution_invalid"]) > 0).astype(int),
    )
    for topic, col in TOPIC_COUNTS.items():
        s[col] = (s["topic"] == topic).astype(int)

    g = s.groupby(["customer_id", "billing_month"])
    agg = g.agg(support_count=("ticket_id", "nunique"),
                **{col: (col, "sum") for col in TOPIC_COUNTS.values()},
                support_resolved_count=("is_resolved", "sum"),
                support_score_count=("score_avail", "count"),
                support_score_mean=("score_avail", "mean"),
                support_flagged_count=("flagged", "sum")).reset_index()

    out = keys[["customer_id", "billing_month"]].merge(agg, on=["customer_id", "billing_month"], how="left")

    if quarantine is not None and len(quarantine):
        q = quarantine[quarantine["customer_linked"] == 1]
        qc = q.groupby(["customer_id", "billing_month"]).size().rename("support_quarantined_linked_count").reset_index()
        out = out.merge(qc, on=["customer_id", "billing_month"], how="left")
    else:
        out["support_quarantined_linked_count"] = 0

    count_cols = [c for c in SUPPORT_COLS if c != "support_score_mean"]
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["support_score_mean"] = out["support_score_mean"].astype(float).round(3)  # NaN, если оценок нет
    return out


def build_customer_month(billing: pd.DataFrame, subscribers: pd.DataFrame, support: pd.DataFrame,
                         quarantine: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n_before = len(billing)
    keys = billing[["customer_id", "billing_month", "available_at"]]
    sup_agg = aggregate_support(support, keys, quarantine)
    subs = subscribers[["customer_id", "activation_date", "activation_date_source", "region",
                        "acquisition_channel", "customer_segment"]]
    cm = billing.merge(subs, on="customer_id", how="left", validate="many_to_one")
    cm = cm.merge(sup_agg, on=["customer_id", "billing_month"], how="left", validate="one_to_one")
    cm["history_month_index"] = cm.groupby("customer_id").cumcount() + 1

    matched = support.merge(keys, on=["customer_id", "billing_month"], how="inner")
    control = {
        "billing_rows": n_before,
        "customer_month_rows": len(cm),
        "key_unique": bool(~cm.duplicated(["customer_id", "billing_month"]).any()),
        "charged_sum_equal": bool(np.isclose(billing["charged_rub"].sum(), cm["charged_rub"].sum())),
        "paid_sum_equal": bool(np.isclose(billing["paid_rub"].sum(), cm["paid_rub"].sum())),
        "support_clean_tickets": len(support),
        "tickets_matched_to_billing_row": len(matched),
        "tickets_without_billing_row": len(support) - len(matched),
        "tickets_counted_in_snapshot": int(cm["support_count"].sum()),
        "tickets_available_after_snapshot": int(len(matched) - cm["support_count"].sum()),
        "rows_with_zero_support": int((cm["support_count"] == 0).sum()),
        "share_rows_zero_support": round(float((cm["support_count"] == 0).mean()), 4),
    }
    return cm, control
