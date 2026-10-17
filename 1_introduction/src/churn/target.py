import pandas as pd

from .config import CAMPAIGN_SHARE, HORIZON_MONTHS, OBSERVATION_END


def horizon_end(t) -> pd.Timestamp:
    return pd.Timestamp(t) + pd.DateOffset(months=HORIZON_MONTHS)


def active_base(contracts: pd.DataFrame, t) -> pd.Index:
    t = pd.Timestamp(t)
    act, ter = contracts["activation"], contracts["termination"]
    mask = (act < t) & (ter.isna() | (ter >= t))
    return contracts.index[mask]


def make_target(contracts: pd.DataFrame, t) -> pd.DataFrame:
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
    t = pd.Timestamp(t)
    av = pd.to_datetime(billing["available_at"])
    return billing.loc[av <= t]


def campaign_size(base_size: int) -> int:
    return int(base_size * CAMPAIGN_SHARE)
