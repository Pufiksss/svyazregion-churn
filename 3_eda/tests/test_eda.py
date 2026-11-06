"""Проверки этапа 3: границы T, доступность, знаменатели, воспроизводимость."""
import pandas as pd
import pytest

from eda.analysis import rates
from eda.pipeline import boundary_example, run
from eda.sample import CSV_KW, network_availability


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    a, b = tmp_path_factory.mktemp("a"), tmp_path_factory.mktemp("b")
    return run(output_dir=a, verbose=False), run(output_dir=b, verbose=False), a


def test_boundary_events_on_T_and_horizon_end():
    be = boundary_example().set_index("customer_id")
    assert be.loc["A", "in_base_at_T"] == 1 and be.loc["A", "churn_target"] == 1
    assert be.loc["B", "in_base_at_T"] == 0
    assert be.loc["C", "churn_target"] == 0
    assert be.loc["D", "churn_target"] == 1


def test_late_network_report_unknown_at_T():
    net = pd.DataFrame({"region": ["R", "R"], "measurement_month": pd.to_datetime(["2026-02-01", "2026-02-01"]),
                        "report_available_at": pd.to_datetime(["2026-03-01 00:00", "2026-03-01 00:00"]),
                        "probe_count": [100, 100], "failed_probe_count": [5, 5],
                        "failed_probe_rate": [0.05, 0.05], "download_mbps_p50": [40, 40], "download_mbps_p90": [80, 80]})
    on_time = network_availability(net.iloc[:1], ["2026-03-01"])
    assert on_time["report_known_at_T"].iloc[0] == 1
    late = net.iloc[:1].assign(report_available_at=pd.Timestamp("2026-03-02 12:00"))
    a = network_availability(late, ["2026-03-01"])
    assert a["report_known_at_T"].iloc[0] == 0 and pd.isna(a["failed_probe_rate_at_T"].iloc[0])


def test_unknown_target_not_in_denominator():
    df = pd.DataFrame({"customer_id": ["a", "b", "c"], "g": ["x", "x", "x"],
                       "churn_target": pd.array([1, 0, pd.NA], dtype="Int64")})
    r = rates(df, "g")
    assert r["n_observations"].iloc[0] == 2 and r["churn_rate"].iloc[0] == 0.5


def test_all_checks_pass(runs):
    assert runs[0]["checks"]["passed"].all(), runs[0]["checks"]


def test_reproducible(runs):
    assert runs[0]["summary"]["csv_sha256"] == runs[1]["summary"]["csv_sha256"]


def test_group_tables_reconstruct_from_sample(runs):
    out = runs[2]
    s = pd.read_csv(out / "eda_sample.csv", **CSV_KW)
    e = s[s["split"] == "eda"]
    debt = pd.read_csv(out / "tables" / "fig4_churn_by_debt.csv", **CSV_KW)
    for _, r in debt.iterrows():
        g = e[e["debt_group"] == r["debt_group"]]
        assert len(g) == int(r["n_observations"])
        assert (g["churn_target"] == "1").sum() == int(r["n_churn"])
        assert g["customer_id"].nunique() == int(r["n_customers"])


def test_no_validation_test_in_eda_tables(runs):
    dyn = pd.read_csv(runs[2] / "tables" / "fig3_churn_by_T.csv", **CSV_KW)
    assert dyn["snapshot_at"].str[:7].max() == "2026-07"
