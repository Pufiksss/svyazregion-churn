"""Общие правила разбора форматов. Единое время проекта — Москва (UTC+3, без перехода на летнее время)."""
from __future__ import annotations

import pandas as pd

TZ = "Europe/Moscow"
DT_FMT = "%Y-%m-%d %H:%M:%S"          # формат дат-времени во всех подготовленных таблицах (МСК)
DATE_FMT = "%Y-%m-%d"

LEGACY_DT_FMT = "%d.%m.%Y %H:%M:%S"
ISO_Z_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$"
ISO_NAIVE_RE = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$"
LEGACY_RE = r"^\d{2}\.\d{2}\.\d{4} \d{2}:\d{2}:\d{2}$"
CLEAN_DT_RE = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
NUM_RE = r"^-?\d+(\.\d+)?$"

TOPIC_RU = {"Качество связи": "network", "Оплата и начисления": "billing",
            "Тариф": "tariff", "Дополнительные услуги": "services"}
CHANNEL_RU = {"Телефон": "phone", "Чат": "chat", "Офис": "office"}
TOPICS = set(TOPIC_RU.values())
CHANNELS = set(CHANNEL_RU.values())
TARIFF_MAP = {"basic": "Basic", "comfort": "Comfort", "premium": "Premium", "премиум": "Premium"}


def blank(s: pd.Series) -> pd.Series:
    return s.fillna("").astype(str).str.strip() == ""


def parse_legacy(s: pd.Series) -> pd.Series:
    """DD.MM.YYYY HH:MM:SS, московское время -> naive МСК."""
    return pd.to_datetime(s.where(~blank(s)), format=LEGACY_DT_FMT, errors="coerce")


def parse_current(s: pd.Series) -> tuple[pd.Series, pd.Series]:
    """ISO 8601: с суффиксом Z — UTC, без зоны — Москва (подтверждено заказчиком).

    Возвращает naive МСК и маску значений, для которых зона принята по договорённости.
    """
    s = s.fillna("").astype(str).str.strip()
    is_z = s.str.match(ISO_Z_RE)
    is_naive = s.str.match(ISO_NAIVE_RE)
    utc = pd.to_datetime(s.where(is_z), format="%Y-%m-%dT%H:%M:%SZ", errors="coerce", utc=True)
    msk_from_utc = utc.dt.tz_convert(TZ).dt.tz_localize(None)
    msk_naive = pd.to_datetime(s.where(is_naive), format="%Y-%m-%dT%H:%M:%S", errors="coerce")
    return msk_from_utc.fillna(msk_naive), is_naive


def parse_clean(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.where(~blank(s)), format=DT_FMT, errors="coerce")


def parse_month(s: pd.Series) -> pd.Series:
    """billing_month: YYYY-MM-DD или DD.MM.YYYY."""
    s = s.fillna("").astype(str).str.strip()
    iso = pd.to_datetime(s, format=DATE_FMT, errors="coerce")
    ru = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(ru)


def to_num(s: pd.Series) -> pd.Series:
    """Число из строки; десятичная запятая допускается, пустое -> NaN."""
    s = s.fillna("").astype(str).str.strip().str.replace(",", ".", regex=False)
    return pd.to_numeric(s.where(s != ""), errors="coerce")


def fmt_dt(s: pd.Series) -> pd.Series:
    return s.dt.strftime(DT_FMT).where(s.notna(), "")


def fmt_date(s: pd.Series) -> pd.Series:
    return s.dt.strftime(DATE_FMT).where(s.notna(), "")


def contract_dates(events: pd.DataFrame) -> pd.DataFrame:
    """activation / termination на абонента по журналу событий (строки)."""
    ev = events.assign(d=pd.to_datetime(events["event_date"], format=DATE_FMT, errors="coerce"),
                       customer_id=events["customer_id"].str.strip(),
                       event_type=events["event_type"].str.strip())
    out = ev.pivot_table(index="customer_id", columns="event_type", values="d", aggfunc="min")
    out = out.reindex(columns=["activation", "termination"])
    out.columns.name = None
    return out
