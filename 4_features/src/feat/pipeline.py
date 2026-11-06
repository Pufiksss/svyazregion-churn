"""Единый запуск этапа 4: признаки, метки, разбиение, аудит, паспорт, преобразователь, baseline, проверки."""
from __future__ import annotations

import copy
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from . import build as B
from .catalog import catalog
from .prep import CATEGORICAL, NUMERIC, make_preprocessor


def _w(df: pd.DataFrame, path: Path):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(out[c].notna(), "")
    out.to_csv(path, sep=";", index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.6g")


def verify_raw(cfg):
    d = cfg["new_data_dir"]
    for line in (d / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        h, name = line.split(maxsplit=1)
        if hashlib.sha256((d / name.strip()).read_bytes()).hexdigest() != h:
            raise RuntimeError(f"Исходник изменён: {name}")


def baseline_by_date(features: pd.DataFrame, labels: pd.DataFrame, frac: float) -> pd.DataFrame:
    """Сортировка по debt_prev_rub ↓, при равенстве customer_id ↑; неизвестный долг — после известных.
    K = floor(frac × N_active) от всей активной базы."""
    df = features[["customer_id", "snapshot_at", "debt_prev_rub"]].merge(
        labels[["customer_id", "snapshot_at", "churn_target"]], on=["customer_id", "snapshot_at"], validate="one_to_one")
    rows = []
    for T, g in df.groupby("snapshot_at"):
        g = g.assign(unknown=g["debt_prev_rub"].isna().astype(int), neg_debt=-g["debt_prev_rub"].fillna(0))
        g = g.sort_values(["unknown", "neg_debt", "customer_id"], kind="mergesort")
        n = len(g)
        k = int(np.floor(frac * n))
        top = g.head(k)
        pos = int(g["churn_target"].sum())
        hit = int(top["churn_target"].sum())
        rows.append(dict(snapshot_at=T, n_active=n, K=k, n_positive=pos, tp_at_K=hit,
                         recall_at_K=hit / pos if pos else np.nan, precision_at_K=hit / k if k else np.nan,
                         n_unknown_debt=int(g["unknown"].sum()), n_debt_positive=int((g["debt_prev_rub"] > 0).sum()),
                         debt_at_K_border=float(top["debt_prev_rub"].iloc[-1]) if k else np.nan))
    t = pd.DataFrame(rows)
    tot = dict(snapshot_at="ИТОГО train (micro: Σ числителей / Σ знаменателей)", n_active=t.n_active.sum(), K=t.K.sum(),
               n_positive=t.n_positive.sum(), tp_at_K=t.tp_at_K.sum(),
               recall_at_K=t.tp_at_K.sum() / t.n_positive.sum(), precision_at_K=t.tp_at_K.sum() / t.K.sum(),
               n_unknown_debt=t.n_unknown_debt.sum(), n_debt_positive=t.n_debt_positive.sum(), debt_at_K_border=np.nan)
    macro = dict(snapshot_at="СРЕДНЕЕ по датам (macro)", recall_at_K=t.recall_at_K.mean(),
                 precision_at_K=t.precision_at_K.mean())
    t["snapshot_at"] = t["snapshot_at"].dt.strftime("%Y-%m-%d")
    return pd.concat([t, pd.DataFrame([tot, macro])], ignore_index=True).round(5)


def independent_base(cfg, dates) -> dict:
    subs = B.read(cfg["prepared_dir"] / "subscribers_clean.csv")
    ev = B.read(cfg["prepared_dir"] / "contract_events_clean.csv")
    ter = dict(zip(ev.loc[ev.event_type == "termination", "customer_id"], ev.loc[ev.event_type == "termination", "event_date"]))
    out = {}
    for T in dates:
        t = T.strftime("%Y-%m-%d")
        out[T] = sum(1 for c, a in zip(subs.customer_id, subs.activation_date) if a < t and (c not in ter or ter[c] >= t))
    return out


def _same(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True), check_dtype=False)
        return True
    except AssertionError:
        return False


def future_mutation(data: dict, T: pd.Timestamp) -> dict:
    """Портит всё, что не должно быть известно на T: будущие месяцы, непубликованные строки, будущие обращения и метки."""
    d = copy.deepcopy(data)
    b = d["billing"]
    fut = b["available_at"] > T
    b.loc[fut, ["closing_debt_rub", "traffic_gb", "charged_rub", "paid_rub"]] = 99999.0
    b.loc[fut, "tariff"] = "FUTURE"
    s = d["support"]
    extra = s.head(50).copy()
    extra["ticket_id"] = "F" + extra["ticket_id"]
    extra["created_at"] = T - pd.Timedelta(days=3)
    extra["available_at"] = T + pd.Timedelta(seconds=1)
    s = s.copy()
    s.loc[s["resolution_available_at"] > T, "resolved_at"] = T - pd.Timedelta(days=1)
    s.loc[s["created_at"] >= T, "topic"] = "billing"
    d["support"] = pd.concat([s, extra], ignore_index=True)
    g = d["digital"]
    fut = g["available_at"] > T
    g.loc[fut, ["app_sessions", "payment_attempts", "payment_failures", "observed_days"]] = 77
    g.loc[fut, "observation_status"] = "observed"
    ev = d["events"]
    later = (ev["event_type"] == "termination") & (ev["event_date"] >= T)
    ev.loc[later, "event_date"] = ev.loc[later, "event_date"] + pd.Timedelta(days=45)
    return d


def run_checks(data, cfg, features, labels, split, audit, prep_info) -> pd.DataFrame:
    rows = []
    def chk(name, ok, detail=""):
        rows.append(dict(check=name, passed=bool(ok), detail=str(detail)))

    dates = B.all_dates(cfg)
    chk("ключ (customer_id, snapshot_at) уникален во всех слоях",
        all(not x.duplicated(["customer_id", "snapshot_at"]).any() for x in (features, labels, split, audit)), len(features))
    keys_equal = all((x[["customer_id", "snapshot_at"]].values == features[["customer_id", "snapshot_at"]].values).all()
                     for x in (labels, split, audit))
    chk("порядок ключей одинаков в features/labels/split/audit", keys_equal)
    ind = independent_base(cfg, dates)
    size = features.groupby("snapshot_at").size()
    mism = {str(k.date()): (int(size[k]), v) for k, v in ind.items() if size[k] != v}
    chk("строк на T = независимо рассчитанная активная база", not mism, mism or "0 расхождений")

    T = pd.Timestamp("2026-04-01")
    f0, _ = B.build_snapshot(T, data, cfg)
    f1, _ = B.build_snapshot(T, future_mutation(data, T), cfg)
    chk("признаки на T не меняются при изменении будущих записей и меток", _same(f0, f1), f"T = {T.date()}, {len(f0)} строк")

    d = data["digital"]
    cid = f0.loc[f0["digital_status_prev"] == "not_published_at_T", "customer_id"].iloc[0]
    m = (d["customer_id"] == cid) & (d["usage_month"] == T - B.MS)
    dd = copy.deepcopy(data); dd["digital"].loc[m, "available_at"] = T
    on = B.build_snapshot(T, dd, cfg)[0].set_index("customer_id").loc[cid, "digital_status_prev"]
    dd["digital"].loc[m, "available_at"] = T + pd.Timedelta(seconds=1)
    late = B.build_snapshot(T, dd, cfg)[0].set_index("customer_id").loc[cid, "digital_status_prev"]
    chk("цифровой отчёт ровно в T включён, через 1 секунду — нет", on != "not_published_at_T" and late == "not_published_at_T",
        f"{cid}: {on} / {late}")

    fy = features.merge(audit[["customer_id", "snapshot_at", "short_history_lt3"]], on=["customer_id", "snapshot_at"])
    short = fy[fy["n_billing_months_3m"] < 3]
    tenure_months = np.ceil(short["tenure_days"] / 31)
    chk("короткая история не превращается в 3 месяца нулей",
        (short["n_billing_months_3m"] <= tenure_months + 1).all() and (short["debt_months_3m"] <= short["n_billing_months_3m"]).all(),
        f"строк с <3 месяцами: {len(short)}")
    one = features[features["n_billing_months_3m"] == 1]
    chk("при одном месяце истории разности неизвестны", one["debt_change_1m_rub"].isna().all() and one["traffic_change_1m_gb"].isna().all(),
        f"{len(one)} строк")
    tr = features["traffic_change_1m_gb"].notna() & features["traffic_prev_gb"].isna()
    chk("пропущенный месяц не подменяется предыдущим в разности", not tr.any())
    obs = features["digital_status_prev"] == "observed"
    zero_att = obs & (features["payment_attempts_prev"] == 0)
    chk("0 попыток оплаты → доля ошибок неизвестна", features.loc[zero_att, "payment_fail_rate_prev"].isna().all(),
        f"{int(zero_att.sum())} строк")
    chk("not_enabled / export_gap / поздний отчёт не становятся нулевыми сессиями",
        features.loc[~obs, ["app_sessions_prev", "payment_attempts_prev"]].isna().all().all(),
        features.loc[~obs, "digital_status_prev"].value_counts().to_dict())
    chk("метка не входит в признаки", not ({"churn_target", "target_end_exclusive"} & set(features.columns)))
    for k, v in prep_info.items():
        chk(k, v[0], v[1])
    return pd.DataFrame(rows)


def run(config_path=None, output_dir=None, verbose=True) -> dict:
    cfg = B.load_config(config_path)
    if output_dir is not None:
        cfg["output_dir"] = Path(output_dir)
    out = cfg["output_dir"]; out.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    verify_raw(cfg)
    data = B.load_data(cfg)

    dg = data["digital"]
    src_checks = {
        "digital_key_unique": not dg.duplicated(["customer_id", "usage_month"]).any(),
        "digital_refs_ok": set(dg.customer_id) <= set(data["subscribers"].customer_id),
        "digital_status_domain_ok": set(dg.observation_status) <= {"observed", "not_enabled", "export_gap"},
        "digital_pairs_equal_billing": set(zip(dg.customer_id, dg.usage_month)) == set(zip(data["billing"].customer_id, data["billing"].billing_month)),
        "digital_failures_le_attempts": bool((dg.payment_failures <= dg.payment_attempts)[dg.observation_status == "observed"].all()),
        "digital_available_after_month_end": bool((dg.available_at >= dg.usage_month + B.MS).all()),
        "digital_nonobserved_empty": bool(dg.loc[dg.observation_status != "observed", ["app_sessions", "observed_days"]].isna().all().all()),
    }

    features, labels, split, audit = B.build_all(data, cfg)
    _w(features, out / "features.csv"); _w(labels, out / "labels.csv")
    _w(split, out / "split.csv"); _w(audit, out / "feature_audit.csv")
    say(f"[1] features: {features.shape}, labels/split/audit согласованы по ключам")

    cat = catalog(NUMERIC + CATEGORICAL)
    _w(cat, out / "feature_catalog.csv")

    tr = split["split"] == "train"
    va = split["split"] == "validation"
    te = split["split"] == "test"
    prep = make_preprocessor()
    Xtr = prep.fit_transform(features.loc[tr, NUMERIC + CATEGORICAL])
    Xva = prep.transform(features.loc[va, NUMERIC + CATEGORICAL])
    Xte = prep.transform(features.loc[te, NUMERIC + CATEGORICAL])
    names = list(prep.get_feature_names_out())
    med = dict(zip(NUMERIC, prep.named_transformers_["numeric"].named_steps["impute"].statistics_))
    med_train = features.loc[tr, NUMERIC].median()
    va_new = features.loc[va, NUMERIC + CATEGORICAL].copy()
    va_new.iloc[0, va_new.columns.get_loc("tariff_prev")] = "Ultra_NEW"
    Xnew = prep.transform(va_new)
    ohe = [i for i, n in enumerate(names) if n.startswith("categorical__tariff_prev_")]
    prep_info = {
        "медианы преобразователя = медианы train": (all(np.isclose(med[c], med_train[c]) for c in NUMERIC), "fit только на train"),
        "новая категория в validation → нули в One-Hot, без ошибки": (bool((Xnew[0, ohe] == 0).all()) and Xnew.shape == Xva.shape, "Ultra_NEW"),
        "transform сохраняет число и порядок строк": (Xva.shape[0] == int(va.sum()) and Xte.shape[0] == int(te.sum()), f"{Xva.shape}, {Xte.shape}"),
    }
    pd.DataFrame({"position": range(len(names)), "x_column": names}).to_csv(out / "x_columns.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.DataFrame([{"part": p, "rows": m.shape[0], "x_columns": m.shape[1]} for p, m in [("train", Xtr), ("validation", Xva), ("test", Xte)]]
                 ).to_csv(out / "x_shapes.csv", sep=";", index=False, encoding="utf-8-sig")
    pd.Series(med).rename("train_median").to_csv(out / "train_medians.csv", sep=";", encoding="utf-8-sig")
    say(f"[2] X: train {Xtr.shape}, validation {Xva.shape}, test {Xte.shape}")

    base = baseline_by_date(features[tr], labels[tr], cfg["campaign_fraction"])
    _w(base, out / "baseline_train.csv")
    say(f"[3] baseline (train, micro): Recall@K = {base.iloc[-2]['recall_at_K']:.3f}, Precision@K = {base.iloc[-2]['precision_at_K']:.3f}")

    checks = run_checks(data, cfg, features, labels, split, audit, prep_info)
    for k, v in src_checks.items():
        checks.loc[len(checks)] = [f"источник: {k}", v, ""]
    _w(checks, out / "checks.csv")

    parts = split.merge(labels, on=["customer_id", "snapshot_at"]).groupby("split").agg(
        dates=("snapshot_at", "nunique"), rows=("customer_id", "size"), customers=("customer_id", "nunique"))
    _w(parts.reset_index(), out / "split_summary.csv")
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(out.glob("*.csv"))}
    summary = dict(python=platform.python_version(), pandas=pd.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
                   config={k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items() if k not in ("prepared_dir", "new_data_dir", "output_dir")},
                   n_numeric=len(NUMERIC), n_categorical=len(CATEGORICAL), x_width=len(names),
                   n_content_features=int((cat["kind"] == "content").sum()),
                   split=parts.reset_index().to_dict("records"), checks_passed=bool(checks["passed"].all()), csv_sha256=hashes)
    (out / "features_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    say(f"[4] проверки: {'все пройдены' if summary['checks_passed'] else 'ЕСТЬ ОШИБКИ'} ({len(checks)})")
    say(f"Готово: {out}")
    return {"summary": summary, "checks": checks, "baseline": base}
