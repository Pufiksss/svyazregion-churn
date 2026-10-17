import pandas as pd
import pytest

from churn.load import load_raw, contract_dates
from churn.target import active_base, make_target, billing_available_at, campaign_size


@pytest.fixture(scope="module")
def data():
    subs, billing, events = load_raw()
    return subs, billing, events, contract_dates(events)


def test_one_activation_per_customer(data):
    subs, _, events, contracts = data
    assert events["event_id"].is_unique
    assert contracts["activation"].notna().all()
    assert len(contracts) == subs["customer_id"].nunique()


def test_base_excludes_terminated_and_future(data):
    *_, contracts = data
    t = pd.Timestamp("2026-09-01")
    base = contracts.loc[active_base(contracts, t)]
    assert (base["activation"] < t).all()
    assert (base["termination"].isna() | (base["termination"] >= t)).all()


def test_target_window(data):
    *_, contracts = data
    y = make_target(contracts, "2026-09-01")
    assert len(y) == 1916 and y["churn"].sum() == 65
    assert make_target(contracts, "2026-10-01")["churn"].isna().all()


def test_no_billing_from_future(data):
    _, billing, *_ = data
    t = pd.Timestamp("2026-09-01")
    av = pd.to_datetime(billing_available_at(billing, t)["available_at"])
    assert (av <= t).all()


def test_campaign_limit():
    assert campaign_size(1916) == 191
