"""Признаки на дату расчёта T: ключ (customer_id, snapshot_at).

Правила (устав этапа 1, config.json):
* T — 00:00 МСК 1-го числа; база — перед событиями на T: activation < T и (нет termination или termination >= T).
* Каждая строка любого источника используется, только если её момент доступности <= T.
* «Предыдущий месяц» — строго T − 1 мес.; окно 3 мес. — календарные месяцы [T − 3 мес., T).
* Метка строится отдельно (labels) и в признаках не участвует.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CSV_KW = dict(sep=";", encoding="utf-8-sig", dtype=str, keep_default_na=False)
MS = pd.offsets.MonthBegin(1)


def load_config(path: Path | None = None) -> dict:
    cfg = json.loads((path or ROOT / "config.json").read_text(encoding="utf-8"))
    for key in ("prepared_dir", "new_data_dir", "output_dir"):
        p = Path(cfg[key])
        cfg[key] = p if p.is_absolute() else (ROOT / p).resolve()
    return cfg


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, **CSV_KW)


def _num(s):
    return pd.to_numeric(s.replace("", np.nan))


def _msk(s):
    return pd.to_datetime(s, utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)


def load_data(cfg: dict) -> dict[str, pd.DataFrame]:
    P = cfg["prepared_dir"]
    subs = read(P / "subscribers_clean.csv")
    subs["activation_date"] = pd.to_datetime(subs["activation_date"])
    ev = read(P / "contract_events_clean.csv")
    ev["event_date"] = pd.to_datetime(ev["event_date"])

    cm = read(P / "customer_month.csv")
    cm["billing_month"] = pd.to_datetime(cm["billing_month"])
    cm["available_at"] = pd.to_datetime(cm["available_at"])
    for c in ["charged_rub", "paid_rub", "closing_debt_rub", "traffic_gb", "voice_minutes", "active_days",
              "payment_delay_days", "tariff_changes_count"]:
        cm[c] = _num(cm[c])
    cm = cm[["customer_id", "billing_month", "available_at", "tariff", "charged_rub", "paid_rub",
             "closing_debt_rub", "traffic_gb", "voice_minutes", "active_days", "payment_delay_days",
             "tariff_changes_count"]]

    sup = read(P / "support_clean.csv")
    for c in ["created_at", "available_at", "resolved_at", "resolution_available_at"]:
        sup[c] = pd.to_datetime(sup[c].replace("", None))
    sup = sup[["ticket_id", "customer_id", "created_at", "available_at", "topic", "resolved_at",
               "resolution_available_at"]]

    dig = read(cfg["new_data_dir"] / "digital_usage_monthly.csv")
    dig["usage_month"] = pd.to_datetime(dig["usage_month"])
    dig["available_at"] = _msk(dig["available_at"])
    for c in ["observed_days", "app_sessions", "payment_attempts", "payment_failures", "selfservice_actions"]:
        dig[c] = _num(dig[c])
    return {"subscribers": subs, "events": ev, "billing": cm, "support": sup, "digital": dig}


def contracts(events: pd.DataFrame) -> pd.DataFrame:
    out = events.pivot_table(index="customer_id", columns="event_type", values="event_date", aggfunc="min")
    out = out.reindex(columns=["activation", "termination"]).apply(pd.to_datetime)
    out.columns.name = None
    return out


def active_base(con: pd.DataFrame, T, mode: str = "before_events") -> pd.Index:
    T = pd.Timestamp(T)
    if mode == "before_events":
        m = (con["activation"] < T) & (con["termination"].isna() | (con["termination"] >= T))
    elif mode == "after_events":
        m = (con["activation"] <= T) & (con["termination"].isna() | (con["termination"] > T))
    else:
        raise ValueError(mode)
    return con.index[m].sort_values()


def horizon_end(T, horizon: str):
    T = pd.Timestamp(T)
    return T + MS if horizon == "calendar_month" else T + pd.Timedelta(days=30)


def build_labels(T, data: dict, cfg: dict) -> pd.DataFrame:
    T = pd.Timestamp(T)
    con = contracts(data["events"])
    ids = active_base(con, T, cfg["active_mode"])
    end = horizon_end(T, cfg["horizon"])
    ter = con.loc[ids, "termination"]
    if end > pd.Timestamp(cfg["observation_end_exclusive"]):
        y = pd.Series(pd.NA, index=ids, dtype="Int64")
    else:
        y = ((ter >= T) & (ter < end)).astype("Int64")
    return pd.DataFrame({"customer_id": ids, "snapshot_at": T, "churn_target": y.values,
                         "target_end_exclusive": end})


def _ratio(num, den):
    """Нулевой или неизвестный знаменатель -> NaN."""
    num, den = pd.Series(num, dtype=float), pd.Series(den, dtype=float)
    return num / den.where(den > 0)


def billing_features(ids, T, cm: pd.DataFrame, W: int) -> tuple[pd.DataFrame, pd.Series]:
    m1, m2, w0 = T - MS, T - 2 * MS, T - W * MS
    b = cm[cm["customer_id"].isin(ids) & (cm["available_at"] <= T) & (cm["billing_month"] >= w0)
           & (cm["billing_month"] < T)]
    p1 = b[b["billing_month"] == m1].set_index("customer_id")
    p2 = b[b["billing_month"] == m2].set_index("customer_id")
    g = b.groupby("customer_id")
    tk = b[b["traffic_gb"].notna()].groupby("customer_id")

    f = pd.DataFrame(index=ids)
    f["tariff_prev"] = p1["tariff"]
    f["debt_prev_rub"] = p1["closing_debt_rub"]
    f["payment_delay_prev_days"] = p1["payment_delay_days"]
    f["traffic_prev_gb"] = p1["traffic_gb"]
    f["voice_prev_min"] = p1["voice_minutes"]
    f["debt_change_1m_rub"] = p1["closing_debt_rub"] - p2["closing_debt_rub"]
    f["traffic_change_1m_gb"] = p1["traffic_gb"] - p2["traffic_gb"]
    f["n_billing_months_3m"] = g.size()
    f["debt_months_3m"] = g["closing_debt_rub"].apply(lambda s: (s > 0).sum())
    f["pay_to_charge_ratio_3m"] = _ratio(g["paid_rub"].sum(), g["charged_rub"].sum())
    f["traffic_per_active_day_3m"] = _ratio(tk["traffic_gb"].sum(), tk["active_days"].sum())
    f["n_traffic_known_3m"] = tk.size()
    f["tariff_changes_3m"] = g["tariff_changes_count"].sum()
    for c in ["n_billing_months_3m", "n_traffic_known_3m"]:
        f[c] = f[c].fillna(0).astype(int)
    none = f["n_billing_months_3m"] == 0
    f.loc[none, ["debt_months_3m", "tariff_changes_3m"]] = np.nan
    max_av = g["available_at"].max().reindex(ids)
    return f, max_av


def support_features(ids, T, sup: pd.DataFrame, W: int) -> tuple[pd.DataFrame, pd.Series]:
    """Пересчёт на T: created_at в [T − 3 мес., T) и available_at <= T (включая поздно поступившие)."""
    w0 = T - W * MS
    s = sup[sup["customer_id"].isin(ids) & (sup["created_at"] >= w0) & (sup["created_at"] < T)
            & (sup["available_at"] <= T)]
    resolved_known = s["resolution_available_at"].notna() & (s["resolution_available_at"] <= T)
    s = s.assign(is_open=(~resolved_known).astype(int),
                 is_billing=(s["topic"] == "billing").astype(int),
                 is_network=(s["topic"] == "network").astype(int),
                 is_prev=(s["created_at"] >= T - MS).astype(int))
    g = s.groupby("customer_id")
    f = pd.DataFrame(index=ids)
    f["support_3m"] = g.size()
    f["support_prev"] = g["is_prev"].sum()
    f["support_billing_3m"] = g["is_billing"].sum()
    f["support_network_3m"] = g["is_network"].sum()
    f["support_open_at_T"] = g["is_open"].sum()
    f = f.fillna(0).astype(int)
    return f, g["available_at"].max().reindex(ids)


def digital_features(ids, T, dig: pd.DataFrame, W: int, max_age: int) -> tuple[pd.DataFrame, pd.Series]:
    m1, w0 = T - MS, T - W * MS
    d = dig[dig["customer_id"].isin(ids) & (dig["usage_month"] >= T - max_age * MS) & (dig["usage_month"] < T)]
    known = d[d["available_at"] <= T]
    prev_all = d[d["usage_month"] == m1].set_index("customer_id")
    prev = known[known["usage_month"] == m1].set_index("customer_id")

    f = pd.DataFrame(index=ids)
    status = prev["observation_status"].reindex(ids)
    status = status.where(status.notna(), np.where(prev_all.reindex(ids)["available_at"].notna(),
                                                   "not_published_at_T", "no_row"))
    f["digital_status_prev"] = status
    obs = prev[prev["observation_status"] == "observed"]
    f["app_sessions_prev"] = obs["app_sessions"]
    f["payment_attempts_prev"] = obs["payment_attempts"]
    f["payment_fail_rate_prev"] = _ratio(obs["payment_failures"], obs["payment_attempts"])
    f["sessions_per_observed_day_prev"] = _ratio(obs["app_sessions"], obs["observed_days"])
    f["selfservice_prev"] = obs["selfservice_actions"]

    ko = known[(known["observation_status"] == "observed") & (known["usage_month"] >= w0)]
    f["digital_observed_months_3m"] = ko.groupby("customer_id").size().reindex(ids).fillna(0).astype(int)
    f["app_sessions_per_day_3m"] = _ratio(ko.groupby("customer_id")["app_sessions"].sum(),
                                          ko.groupby("customer_id")["observed_days"].sum())
    last = ko.sort_values("usage_month").groupby("customer_id").tail(1).set_index("customer_id")
    f["digital_last_obs_month"] = last["usage_month"]
    f["digital_last_obs_age_months"] = ((T.year - last["usage_month"].dt.year) * 12
                                        + T.month - last["usage_month"].dt.month)
    f["digital_last_obs_sessions"] = last["app_sessions"]
    return f, known.groupby("customer_id")["available_at"].max().reindex(ids)


def build_snapshot(T, data: dict, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    T = pd.Timestamp(T)
    W = cfg["window_months"]
    con = contracts(data["events"])
    ids = active_base(con, T, cfg["active_mode"])
    subs = data["subscribers"].set_index("customer_id")

    base = pd.DataFrame(index=ids)
    base["tenure_days"] = (T - subs.loc[ids, "activation_date"]).dt.days
    for c in ["region", "customer_segment", "acquisition_channel"]:
        base[c] = subs.loc[ids, c]
    bf, b_av = billing_features(ids, T, data["billing"], W)
    sf, s_av = support_features(ids, T, data["support"], W)
    df_, d_av = digital_features(ids, T, data["digital"], W, cfg["digital_last_report_max_age_months"])
    hist = data["billing"][data["billing"]["customer_id"].isin(ids) & (data["billing"]["available_at"] <= T)]

    feats = pd.concat([base, bf, sf, df_.drop(columns=["digital_last_obs_month"])], axis=1)
    feats["history_months_available"] = hist.groupby("customer_id").size().reindex(ids).fillna(0).astype(int)
    feats.index.name = "customer_id"
    feats = feats.reset_index()
    feats.insert(1, "snapshot_at", T)

    audit = pd.DataFrame({
        "customer_id": ids, "snapshot_at": T, "prev_month": T - MS, "window_start": T - W * MS,
        "window_end_exclusive": T,
        "max_billing_available_at": b_av.values, "max_support_available_at": s_av.values,
        "max_digital_available_at": d_av.values,
        "n_billing_months_3m": bf["n_billing_months_3m"].values,
        "n_traffic_known_3m": bf["n_traffic_known_3m"].values,
        "history_months_available": feats["history_months_available"].values,
        "digital_status_prev": df_["digital_status_prev"].values,
        "digital_last_obs_month": df_["digital_last_obs_month"].values,
        "short_history_lt3": (bf["n_billing_months_3m"] < W).astype(int).values,
    })
    return feats, audit


def split_of(T, cfg) -> str:
    t = pd.Timestamp(T).strftime("%Y-%m-%d")
    for name, key in [("train", "train_dates"), ("validation", "validation_dates"), ("test", "test_dates")]:
        if t in cfg[key]:
            return name
    raise ValueError(t)


def all_dates(cfg) -> list[pd.Timestamp]:
    return [pd.Timestamp(d) for d in cfg["train_dates"] + cfg["validation_dates"] + cfg["test_dates"]]


def build_all(data: dict, cfg: dict):
    F, A, L, S = [], [], [], []
    for T in all_dates(cfg):
        f, a = build_snapshot(T, data, cfg)
        F.append(f); A.append(a)
        L.append(build_labels(T, data, cfg))
        S.append(pd.DataFrame({"customer_id": f["customer_id"], "snapshot_at": T, "split": split_of(T, cfg)}))
    return tuple(pd.concat(x, ignore_index=True) for x in (F, L, S, A))
