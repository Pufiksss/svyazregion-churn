"""Активная база, целевая переменная и отсечение по доступности данных.

Определения (см. docs/project_charter.md):
* дата расчёта t — 00:00 МСК первого дня месяца;
* активная база на t — activation < t и (termination отсутствует или termination >= t);
* отток (y=1) — событие termination с event_date в [t, t + 1 месяц);
* признаки — только строки биллинга с available_at <= t и постоянные атрибуты абонента.
"""
import pandas as pd

from .config import HORIZON_MONTHS, OBSERVATION_END, CAMPAIGN_SHARE


def horizon_end(t) -> pd.Timestamp:
    return pd.Timestamp(t) + pd.DateOffset(months=HORIZON_MONTHS)


def active_base(contracts: pd.DataFrame, t) -> pd.Index:
    """customer_id абонентов, у которых договор действует в 00:00 даты t."""
    t = pd.Timestamp(t)
    act, ter = contracts["activation"], contracts["termination"]
    mask = (act < t) & (ter.isna() | (ter >= t))
    return contracts.index[mask]


def make_target(contracts: pd.DataFrame, t) -> pd.DataFrame:
    """Срез (customer_id, snapshot_date, churn) для активной базы на t.

    Если горизонт выходит за конец наблюдения, churn = NA (исход неизвестен).
    """
    t = pd.Timestamp(t)
    end = horizon_end(t)
    base = active_base(contracts, t)
    ter = contracts.loc[base, "termination"]
    out = pd.DataFrame({"customer_id": base, "snapshot_date": t})
    if end > pd.Timestamp(OBSERVATION_END):
        out["churn"] = pd.NA
    else:
        out["churn"] = ((ter >= t) & (ter < end)).astype(int).values
    return out.reset_index(drop=True)


def billing_available_at(billing: pd.DataFrame, t) -> pd.DataFrame:
    """Строки биллинга, известные на момент t (available_at <= t)."""
    t = pd.Timestamp(t)
    av = pd.to_datetime(billing["available_at"])
    return billing.loc[av <= t]


def campaign_size(base_size: int) -> int:
    """Максимальное число абонентов в кампании (не более 10 % активной базы)."""
    return int(base_size * CAMPAIGN_SHARE)
