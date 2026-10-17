import pandas as pd

from .config import CSV_KW, RAW


def load_raw():
    subscribers = pd.read_csv(RAW / "subscribers.csv", **CSV_KW)
    billing = pd.read_csv(RAW / "billing_monthly.csv", **CSV_KW)
    events = pd.read_csv(RAW / "contract_events.csv", **CSV_KW)
    return subscribers, billing, events


def parse_billing_month(s: pd.Series) -> pd.Series:
    iso = pd.to_datetime(s, format="%Y-%m-%d", errors="coerce")
    ru = pd.to_datetime(s, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(ru)


def contract_dates(events: pd.DataFrame) -> pd.DataFrame:
    ev = events.assign(event_date=pd.to_datetime(events["event_date"]))
    wide = ev.pivot_table(
        index="customer_id",
        columns="event_type",
        values="event_date",
        aggfunc="min",
    )
    wide = wide.reindex(columns=["activation", "termination"])
    wide.columns.name = None
    return wide
