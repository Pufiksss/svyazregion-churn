"""Исследовательская таблица снимков eda_sample: ключ (customer_id, snapshot_at).

Правила (устав этапа 1, config.json):
* T — 00:00 МСК первого числа месяца; база — состояние непосредственно ПЕРЕД событиями на T:
  activation < T и (нет termination или termination >= T). Подключение в день T не входит.
* Признаки — строка customer_month предыдущего календарного месяца, только если её available_at <= T.
* Метка формируется отдельно: churn_target = 1, если termination ∈ [T, target_end_exclusive),
  target_end_exclusive = первое число следующего месяца. Если конец горизонта позже конца наблюдения —
  метка неизвестна (пусто), а не 0.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_KW = dict(sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)


def load_config(path: Path | None = None) -> dict:
    cfg = json.loads((path or ROOT / "config.json").read_text(encoding="utf-8"))
    for key in ("prepared_dir", "new_data_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (ROOT / p).resolve()
    return cfg


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_KW)


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.replace("", np.nan))


def load_prepared(cfg: dict) -> dict[str, pd.DataFrame]:
    P = cfg["prepared_dir"]
    subs = read(P / "subscribers_clean.csv")
    subs["activation_date"] = pd.to_datetime(subs["activation_date"])
    ev = read(P / "contract_events_clean.csv")
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    cm = read(P / "customer_month.csv")
    cm["billing_month"] = pd.to_datetime(cm["billing_month"])
    cm["available_at"] = pd.to_datetime(cm["available_at"])
    for c in ["monthly_fee_rub", "traffic_missing", "voice_minutes_invalid", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb", "voice_minutes",
              "active_days", "payment_delay_days", "tariff_changes_count", "support_count",
              "support_network_count", "support_billing_count", "support_unknown_topic_count",
              "support_resolved_count", "support_score_count", "support_score_mean"]:
        cm[c] = num(cm[c])
    return {"subscribers": subs, "events": ev, "customer_month": cm}


def contracts(events: pd.DataFrame) -> pd.DataFrame:
    out = events.pivot_table(index="customer_id", columns="event_type", values="event_date", aggfunc="min")
    out = out.reindex(columns=["activation", "termination"])
    out.columns.name = None
    return out


def active_base(con: pd.DataFrame, T: pd.Timestamp) -> pd.Index:
    """Состояние перед событиями на T."""
    m = (con["activation"] < T) & (con["termination"].isna() | (con["termination"] >= T))
    return con.index[m]


def horizon_end(T: pd.Timestamp, horizon: str = "calendar_month") -> pd.Timestamp:
    if horizon == "calendar_month":
        return T + pd.offsets.MonthBegin(1)
    if horizon == "30d":
        return T + pd.Timedelta(days=30)
    raise ValueError(horizon)


def make_target(con: pd.DataFrame, ids: pd.Index, T, end, observation_end) -> pd.Series:
    """Метка отдельно от признаков. Неполный горизонт -> NA."""
    if end > pd.Timestamp(observation_end):
        return pd.Series(pd.NA, index=ids, dtype="Int64")
    ter = con.loc[ids, "termination"]
    return ((ter >= T) & (ter < end)).astype("Int64")


FEATURES = ["tariff", "monthly_fee_rub", "charged_rub", "paid_rub", "closing_debt_rub", "payment_delay_days",
            "traffic_gb", "voice_minutes", "active_days", "tariff_changes_count", "traffic_missing",
            "voice_minutes_invalid", "support_count", "support_network_count", "support_billing_count",
            "support_unknown_topic_count"]


def build_snapshot(T, data: dict, cfg: dict, split: str) -> pd.DataFrame:
    T = pd.Timestamp(T)
    con = contracts(data["events"])
    ids = active_base(con, T)
    end = horizon_end(T, cfg["horizon"])
    subs = data["subscribers"].set_index("customer_id")
    cm = data["customer_month"]

    known = cm[(cm["available_at"] <= T) & cm["customer_id"].isin(ids)]
    last_month = T - pd.offsets.MonthBegin(1)
    prev = known[known["billing_month"] == last_month].set_index("customer_id")
    n_hist = known.groupby("customer_id").size()

    s = pd.DataFrame(index=pd.Index(ids, name="customer_id"))
    s["snapshot_at"] = T
    s["split"] = split
    s["source_billing_month"] = prev["billing_month"].reindex(ids)
    s["billing_available_at"] = prev["available_at"].reindex(ids)
    s["has_prev_month"] = s["source_billing_month"].notna().astype(int)
    s["history_months_available"] = n_hist.reindex(ids).fillna(0).astype(int)
    s["region"] = subs.loc[ids, "region"]
    s["acquisition_channel"] = subs.loc[ids, "acquisition_channel"]
    s["customer_segment"] = subs.loc[ids, "customer_segment"]
    s["tenure_days"] = (T - subs.loc[ids, "activation_date"]).dt.days
    for f in FEATURES:
        s[f] = prev[f].reindex(ids)
    s["target_end_exclusive"] = end
    s["churn_target"] = make_target(con, ids, T, end, cfg["observation_end"])
    return s.reset_index()


def add_groups(df: pd.DataFrame, tenure_cut: int = 90) -> pd.DataFrame:
    """Группы задаются заранее; неизвестное значение — отдельная группа."""
    d = df.copy()
    debt = d["closing_debt_rub"]
    d["debt_group"] = np.select([debt.isna(), debt > 0], ["неизвестно", "долг > 0"], "долг = 0")
    sc = d["support_count"]
    d["support_group"] = np.select([sc.isna(), sc == 0, sc == 1], ["нет данных", "0", "1"], "2+")
    d["tenure_group"] = np.where(d["tenure_days"] < tenure_cut, f"<{tenure_cut} дн.", f"≥{tenure_cut} дн.")
    return d


def build_sample(data: dict, cfg: dict) -> pd.DataFrame:
    parts = [build_snapshot(T, data, cfg, split) for split, dates in cfg["splits"].items() for T in dates]
    out = pd.concat(parts, ignore_index=True)
    out = add_groups(out, cfg["tenure_bins_days"][0])
    out["base_rule"] = "activation<T; termination absent or >=T"
    return out.sort_values(["snapshot_at", "customer_id"]).reset_index(drop=True)


def load_network(cfg: dict) -> pd.DataFrame:
    n = read(cfg["new_data_dir"] / "network_quality_monthly.csv")
    n["measurement_month"] = pd.to_datetime(n["measurement_month"])
    n["report_available_at"] = pd.to_datetime(n["report_available_at"], utc=True)\
        .dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    for c in ["probe_count", "failed_probe_count"]:
        n[c] = pd.to_numeric(n[c]).astype(int)
    for c in ["download_mbps_p50", "download_mbps_p90"]:
        n[c] = pd.to_numeric(n[c])
    n["failed_probe_rate"] = n["failed_probe_count"] / n["probe_count"]
    return n


def network_availability(net: pd.DataFrame, snapshots) -> pd.DataFrame:
    """Ключ (region, measurement_month, snapshot_at): месяц измерений = месяц предыдущего биллинга.
    Показатели заполняются только при report_available_at <= snapshot_at."""
    rows = []
    for T in snapshots:
        T = pd.Timestamp(T)
        m = T - pd.offsets.MonthBegin(1)
        part = net[net["measurement_month"] == m].copy()
        part["snapshot_at"] = T
        part["report_known_at_T"] = (part["report_available_at"] <= T).astype(int)
        for c in ["failed_probe_rate", "download_mbps_p50", "download_mbps_p90"]:
            part[c + "_at_T"] = part[c].where(part["report_known_at_T"] == 1)
        rows.append(part)
    cols = ["region", "measurement_month", "snapshot_at", "report_available_at", "report_known_at_T",
            "probe_count", "failed_probe_count", "failed_probe_rate", "failed_probe_rate_at_T",
            "download_mbps_p50_at_T", "download_mbps_p90_at_T"]
    return pd.concat(rows, ignore_index=True)[cols]
