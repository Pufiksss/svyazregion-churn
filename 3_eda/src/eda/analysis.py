from __future__ import annotations

import numpy as np
import pandas as pd

STUDY_FIELDS = ["tariff", "region", "tenure_days", "closing_debt_rub", "payment_delay_days", "charged_rub",
                "traffic_gb", "voice_minutes", "active_days", "support_count", "support_network_count",
                "support_billing_count"]


def rates(df: pd.DataFrame, by, small: int = 30) -> pd.DataFrame:
    by = [by] if isinstance(by, str) else list(by)
    d = df[df["churn_target"].notna()]
    t = d.groupby(by, observed=True).agg(n_observations=("customer_id", "size"),
                                         n_customers=("customer_id", "nunique"),
                                         n_churn=("churn_target", "sum")).reset_index()
    t["n_churn"] = t["n_churn"].astype(int)
    t["churn_rate"] = (t["n_churn"] / t["n_observations"]).round(5)
    t["small_group"] = (t["n_observations"] < small).astype(int)
    return t


def missing_table(df: pd.DataFrame) -> pd.DataFrame:
    reasons = {
        "traffic_gb": "пропуск в источнике биллинга (флаг traffic_missing), не восстанавливается",
        "voice_minutes": "отрицательное значение в источнике заменено пропуском (voice_minutes_invalid)",
    }
    rows = []
    for f in STUDY_FIELDS:
        miss = int(df[f].isna().sum())
        rows.append(dict(field=f, n_rows=len(df), n_known=len(df) - miss, n_missing=miss,
                         missing_share=round(miss / len(df), 5), reason=reasons.get(f, "")))
    return pd.DataFrame(rows)


def distribution(df: pd.DataFrame, fields=("traffic_gb", "closing_debt_rub", "payment_delay_days",
                                           "charged_rub", "voice_minutes", "tenure_days", "support_count")) -> pd.DataFrame:
    rows = []
    for f in fields:
        x = df[f].dropna()
        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
        rows.append(dict(field=f, n_known=len(x), n_missing=int(df[f].isna().sum()), mean=x.mean(),
                         p05=q[0.05], p25=q[0.25], median=q[0.5], p75=q[0.75], p90=q[0.9], p95=q[0.95],
                         p99=q[0.99], max=x.max(), share_zero=(x == 0).mean()))
    return pd.DataFrame(rows).round(3)


def categories(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for f in ["tariff", "region", "customer_segment", "acquisition_channel", "debt_group", "support_group",
              "tenure_group"]:
        vc = df[f].value_counts()
        for k, v in vc.items():
            rows.append(dict(field=f, value=k, n_rows=int(v), share=round(v / len(df), 4)))
    return pd.DataFrame(rows)


def by_snapshot(df: pd.DataFrame, independent_base: dict) -> pd.DataFrame:
    t = rates(df, "snapshot_at")
    t["independent_base_size"] = t["snapshot_at"].map(independent_base)
    return t


def stratified(df: pd.DataFrame, group: str, strata: str, a: str, b: str, small: int = 30) -> pd.DataFrame:
    """Сравнение групп a и b внутри каждого слоя strata (проверка альтернативного объяснения)."""
    t = rates(df[df[group].isin([a, b])], [strata, group], small)
    w = t.pivot(index=strata, columns=group, values=["n_observations", "n_churn", "churn_rate"])
    w.columns = [f"{m}__{g}" for m, g in w.columns]
    w = w.reset_index()
    w["rate_diff"] = (w[f"churn_rate__{a}"] - w[f"churn_rate__{b}"]).round(5)
    w["rate_ratio"] = (w[f"churn_rate__{a}"] / w[f"churn_rate__{b}"]).round(3)
    w["small_group"] = ((w[f"n_observations__{a}"] < small) | (w[f"n_observations__{b}"] < small)).astype(int)
    w.insert(0, "strata", strata)
    w = w.rename(columns={strata: "stratum"})
    w["stratum"] = w["stratum"].astype(str)
    return w


def cluster_bootstrap_diff(df: pd.DataFrame, group: str, a: str, b: str, iters: int, seed: int) -> dict:
    """95 % ДИ разности долей a − b, ресэмплинг абонентов целиком (учёт повторных наблюдений)."""
    d = df[df["churn_target"].notna() & df[group].isin([a, b])]
    g = d.groupby(["customer_id", group])["churn_target"].agg(["sum", "size"]).unstack(fill_value=0)
    s_a, n_a = g[("sum", a)].to_numpy(), g[("size", a)].to_numpy()
    s_b, n_b = g[("sum", b)].to_numpy(), g[("size", b)].to_numpy()
    rng = np.random.default_rng(seed)
    k = len(g)
    diffs = np.empty(iters)
    for i in range(iters):
        idx = rng.integers(0, k, k)
        diffs[i] = s_a[idx].sum() / n_a[idx].sum() - s_b[idx].sum() / n_b[idx].sum()
    point = s_a.sum() / n_a.sum() - s_b.sum() / n_b.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return dict(comparison=f"{group}: {a} − {b}", rate_a=round(s_a.sum() / n_a.sum(), 5),
                rate_b=round(s_b.sum() / n_b.sum(), 5), diff=round(point, 5), ci95_low=round(lo, 5),
                ci95_high=round(hi, 5), n_customers=k, iterations=iters)


def network_monthly(net: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Доля неуспешных проб по региону и месяцу + доступность отчёта на ближайшую T (= месяц + 1)."""
    t = net[["region", "measurement_month", "report_available_at", "probe_count", "failed_probe_count",
             "failed_probe_rate", "download_mbps_p50", "download_mbps_p90"]].copy()
    a = avail[["region", "measurement_month", "snapshot_at", "report_known_at_T"]]
    t = t.merge(a, on=["region", "measurement_month"], how="left")
    t["failed_probe_rate"] = t["failed_probe_rate"].round(5)
    return t.sort_values(["measurement_month", "region"]).reset_index(drop=True)


def network_pooled(net: pd.DataFrame) -> pd.DataFrame:
    g = net.groupby("measurement_month")[["probe_count", "failed_probe_count"]].sum().reset_index()
    g["failed_probe_rate_pooled"] = (g["failed_probe_count"] / g["probe_count"]).round(5)
    g["region"] = "все регионы (сумма/сумма)"
    return g


def churn_by_network(df: pd.DataFrame, avail: pd.DataFrame) -> pd.DataFrame:
    """Только известные на T отчёты. Неизвестный отчёт — отдельная группа."""
    a = avail[["region", "snapshot_at", "report_known_at_T", "failed_probe_rate_at_T"]]
    d = df.merge(a, on=["region", "snapshot_at"], how="left", validate="many_to_one")
    known = d["failed_probe_rate_at_T"]
    cut = known.dropna().median()
    d["network_group"] = np.select([known.isna(), known > cut],
                                   ["отчёт неизвестен на T", f"доля ошибок > {cut:.4f} (медиана)"],
                                   f"доля ошибок ≤ {cut:.4f}")
    return rates(d, "network_group")
