from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from . import analysis as A
from . import figures as F
from .sample import (build_sample, contracts, load_config, load_network, load_prepared, make_target,
                     active_base, network_availability, read)


def _w(df: pd.DataFrame, path: Path):
    out = df.copy()
    for c in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[c]):
            out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S").where(out[c].notna(), "")
    out.to_csv(path, sep=";", index=False, encoding="utf-8-sig", lineterminator="\n", float_format="%.6g")


def verify_raw(cfg: dict) -> bool:
    d = cfg["new_data_dir"]
    for line in (d / "CHECKSUMS.sha256").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if hashlib.sha256((d / name.strip()).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Исходник изменён: {name}")
    return True


def independent_base_sizes(cfg: dict, dates) -> dict:
    subs = read(cfg["prepared_dir"] / "subscribers_clean.csv")
    ev = read(cfg["prepared_dir"] / "contract_events_clean.csv")
    term = dict(zip(ev.loc[ev.event_type == "termination", "customer_id"],
                    ev.loc[ev.event_type == "termination", "event_date"]))
    out = {}
    for T in dates:
        t = pd.Timestamp(T).strftime("%Y-%m-%d")
        out[pd.Timestamp(T)] = sum(1 for cid, a in zip(subs.customer_id, subs.activation_date)
                                   if a < t and (cid not in term or term[cid] >= t))
    return out


def boundary_example() -> pd.DataFrame:
    """Искусственный пример событий ровно на T и ровно на конце горизонта."""
    T, end = pd.Timestamp("2026-03-01"), pd.Timestamp("2026-04-01")
    ev = pd.DataFrame([
        ("A", "activation", "2025-01-10"), ("A", "termination", "2026-03-01"),
        ("B", "activation", "2026-03-01"),
        ("C", "activation", "2025-05-05"), ("C", "termination", "2026-04-01"),
        ("D", "activation", "2025-05-05"), ("D", "termination", "2026-03-31"),
        ("E", "activation", "2026-02-28"),
    ], columns=["customer_id", "event_type", "event_date"])
    ev["event_date"] = pd.to_datetime(ev["event_date"])
    con = contracts(ev)
    base = active_base(con, T)
    y = make_target(con, base, T, end, "2026-10-01")
    res = pd.DataFrame({"customer_id": list("ABCDE")})
    res["in_base_at_T"] = res["customer_id"].isin(base).astype(int)
    res["churn_target"] = res["customer_id"].map(y)
    res["expected_in_base"] = [1, 0, 1, 1, 1]
    res["expected_target"] = [1, pd.NA, 0, 1, 0]
    res["rule"] = ["termination = T: был в базе перед событиями T, ушёл в горизонте",
                   "activation = T: не входит (подключение в день T)",
                   "termination = конец горизонта: конец исключён -> 0 (уход относится к следующему T)",
                   "termination в последний день горизонта -> 1",
                   "activation накануне T: входит, история короткая"]
    return res


HYPOTHESES = [
    ("H1", "Отличается ли будущая доля ухода у абонентов с долгом и без долга?",
     "churn_rate по строкам «абонент × T»", "долг > 0 / долг = 0 / неизвестно (closing_debt_rub прошлого месяца)",
     "T 01.12.2025–01.07.2026", "заранее не задано",
     "долг — состояние на конец месяца, погашение после T не видно; повторные наблюдения одного абонента"),
    ("H2", "Связаны ли доступные к T обращения в поддержку с последующим уходом?",
     "churn_rate", "support_count прошлого месяца: 0 / 1 / 2+ / нет данных",
     "T 01.12.2025–01.07.2026", "заранее не задано",
     "0 = нет доступных корректно связанных обращений; 20 обращений в карантине; смена системы 11.2025"),
    ("H3", "Меняется ли доля оттока и размер активной базы во времени?",
     "churn_rate и размер базы на каждую T", "даты T", "T 01.12.2025–01.07.2026", "заранее не задано",
     "8 дат — мало для сезонности; одна и та же база на соседних T"),
    ("H4", "Различается ли отток по стажу и не объясняется ли это составом групп?",
     "churn_rate", "стаж на T: <90 / ≥90 дней; проверка внутри групп долга и месяцев",
     "T 01.12.2025–01.07.2026", "заранее не задано",
     "группа <90 дней мала (≈1,1 тыс. строк, 40 уходов); стаж и долг могут быть связаны"),
]


def run(config_path: Path | None = None, output_dir: Path | None = None, verbose: bool = True) -> dict:
    cfg = load_config(config_path)
    if output_dir is not None:
        cfg["output_dir"] = Path(output_dir)
    out = cfg["output_dir"]
    tab, fig = out / "tables", out / "figures"
    tab.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)
    say = print if verbose else (lambda *a, **k: None)
    small, seed, iters = cfg["small_group_rows"], cfg["random_seed"], cfg["bootstrap_iterations"]

    verify_raw(cfg)
    data = load_prepared(cfg)
    sample = build_sample(data, cfg)
    _w(sample, out / "eda_sample.csv")
    eda = sample[sample["split"] == "eda"].copy()
    eda_dates = [pd.Timestamp(t) for t in cfg["splits"]["eda"]]
    period = "T 01.12.2025–01.07.2026"
    say(f"[1] eda_sample: {len(sample)} строк, из них EDA {len(eda)}, абонентов EDA {eda['customer_id'].nunique()}")

    miss = A.missing_table(eda)
    dist = A.distribution(eda)
    cats = A.categories(eda)
    for name, t in [("fig1_missing", miss), ("fig2_distributions", dist), ("categories", cats)]:
        _w(t, tab / f"{name}.csv")

    indep_all = independent_base_sizes(cfg, [d for v in cfg["splits"].values() for d in v])
    dyn = A.by_snapshot(eda, indep_all)
    debt = A.rates(eda, "debt_group", small)
    sup = A.rates(eda, "support_group", small)
    ten = A.rates(eda, "tenure_group", small)
    ten_by_debt = A.rates(eda, ["debt_group", "tenure_group"], small)
    extra = {c: A.rates(eda, c, small) for c in ["tariff", "region", "customer_segment"]}
    strat = pd.concat([
        A.stratified(eda.assign(T=eda["snapshot_at"].dt.strftime("%Y-%m")), "debt_group", "T", "долг > 0", "долг = 0", small),
        A.stratified(eda, "debt_group", "tariff", "долг > 0", "долг = 0", small),
        A.stratified(eda, "debt_group", "tenure_group", "долг > 0", "долг = 0", small),
        A.stratified(eda, "tenure_group", "debt_group", "<90 дн.", "≥90 дн.", small),
        A.stratified(eda.assign(T=eda["snapshot_at"].dt.strftime("%Y-%m")), "tenure_group", "T", "<90 дн.", "≥90 дн.", small),
    ], ignore_index=True)
    eda_sup = eda.assign(support_any=np.where(eda["support_count"] > 0, "1+", "0"))
    boot = pd.DataFrame([
        A.cluster_bootstrap_diff(eda, "debt_group", "долг > 0", "долг = 0", iters, seed),
        A.cluster_bootstrap_diff(eda_sup, "support_any", "1+", "0", iters, seed),
        A.cluster_bootstrap_diff(eda, "tenure_group", "<90 дн.", "≥90 дн.", iters, seed),
    ])
    for name, t in [("fig3_churn_by_T", dyn), ("fig4_churn_by_debt", debt), ("fig4_churn_by_support", sup),
                    ("fig5_churn_by_tenure", ten), ("fig5_tenure_by_debt", ten_by_debt),
                    ("stratified_checks", strat), ("bootstrap_ci", boot)] + \
            [(f"extra_churn_by_{k}", v) for k, v in extra.items()]:
        _w(t, tab / f"{name}.csv")

    net = load_network(cfg)
    all_T = [pd.Timestamp(d) for v in cfg["splits"].values() for d in v]
    avail = network_availability(net, all_T)
    _w(avail, out / "network_availability.csv")
    nm = A.network_monthly(net, avail)
    pooled = A.network_pooled(net)
    by_net = A.churn_by_network(eda, avail[avail["snapshot_at"].isin(eda_dates)])
    ctx = read(cfg["new_data_dir"] / "business_context.csv")
    ctx_t = ctx.assign(
        announced_at_msk=pd.to_datetime(ctx["announced_at"], utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None),
        event_end_exclusive=pd.to_datetime(ctx["event_end"]) + pd.Timedelta(days=1))
    ctx_t["announced_after_period_end"] = (ctx_t["announced_at_msk"] >= ctx_t["event_end_exclusive"]).astype(int)
    ctx_t["first_T_known"] = ctx_t["announced_at_msk"].apply(
        lambda a: a if (a.day == 1 and a == a.normalize()) else a.to_period("M").to_timestamp() + pd.offsets.MonthBegin(1))
    for name, t in [("fig6_network_monthly", nm), ("fig6_network_pooled", pooled), ("churn_by_network_at_T", by_net),
                    ("business_context_timing", ctx_t)]:
        _w(t, tab / f"{name}.csv")

    F.fig1_missing(miss, period, fig / "fig1_missing.png")
    F.fig2_distributions(eda, fig / "fig2_distributions.png")
    F.fig3_dynamics(dyn, fig / "fig3_churn_by_T.png")
    F.fig4_debt_support(debt, sup, fig / "fig4_debt_support.png")
    F.fig5_segment(ten, ten_by_debt, fig / "fig5_tenure.png")
    F.fig6_network(nm, pooled, [d - pd.offsets.MonthBegin(1) for d in eda_dates], fig / "fig6_network.png")
    say("[2] таблицы и 6 рисунков сохранены")

    d_r = debt.set_index("debt_group")["churn_rate"]
    s_r = sup.set_index("support_group")["churn_rate"]
    t_r = ten.set_index("tenure_group")["churn_rate"]
    b = boot.set_index("comparison")
    debt_strata = strat[strat["n_observations__долг > 0"].notna()]
    debt_higher = int((debt_strata["rate_diff"] > 0).sum())
    ten_strata = strat[strat["n_observations__<90 дн."].notna()]
    results = {
        "H1": ("поддерживается на EDA-периоде",
               f"долг > 0: {d_r['долг > 0']:.2%} vs долг = 0: {d_r['долг = 0']:.2%}; разность "
               f"{b.loc['debt_group: долг > 0 − долг = 0', 'diff']:.2%}, 95 % ДИ (бутстрэп по абонентам) "
               f"[{b.loc['debt_group: долг > 0 − долг = 0', 'ci95_low']:.2%}; "
               f"{b.loc['debt_group: долг > 0 − долг = 0', 'ci95_high']:.2%}]; выше в {debt_higher} из {len(debt_strata)} слоёв (месяц, тариф, стаж)",
               "рис. 4, fig4_churn_by_debt.csv, stratified_checks.csv, bootstrap_ci.csv",
               "проверить окна долга (сумма, длительность, задержка оплаты) на этапе 4"),
        "H2": ("не поддерживается / связь слабая",
               f"0 обращений: {s_r['0']:.2%}, 1: {s_r['1']:.2%}, 2+: {s_r['2+']:.2%}; разность 1+ − 0 "
               f"{b.loc['support_any: 1+ − 0', 'diff']:.2%}, ДИ [{b.loc['support_any: 1+ − 0', 'ci95_low']:.2%}; "
               f"{b.loc['support_any: 1+ − 0', 'ci95_high']:.2%}] включает 0",
               "рис. 4, fig4_churn_by_support.csv, bootstrap_ci.csv",
               "проверить окна 3 мес., темы billing/network и закрытия без решения"),
        "H3": ("поддерживается: доля растёт, база слегка сокращается",
               f"доля {dyn['churn_rate'].iloc[0]:.2%} ({dyn['n_churn'].iloc[0]}/{dyn['n_observations'].iloc[0]}) на 01.12.2025 → "
               f"{dyn['churn_rate'].iloc[-1]:.2%} ({dyn['n_churn'].iloc[-1]}/{dyn['n_observations'].iloc[-1]}) на 01.07.2026",
               "рис. 3, fig3_churn_by_T.csv",
               "учесть дрейф: валидация по последним датам, признак календарного месяца не использовать как причину"),
        "H4": ("данных недостаточно (направление: выше у <90 дн.)",
               f"<90 дн.: {t_r['<90 дн.']:.2%} vs ≥90 дн.: {t_r['≥90 дн.']:.2%}; ДИ разности "
               f"[{b.loc['tenure_group: <90 дн. − ≥90 дн.', 'ci95_low']:.2%}; {b.loc['tenure_group: <90 дн. − ≥90 дн.', 'ci95_high']:.2%}]; "
               f"выше в {int((ten_strata['rate_diff'] > 0).sum())} из {len(ten_strata)} слоёв (долг, месяц)",
               "рис. 5, fig5_churn_by_tenure.csv, fig5_tenure_by_debt.csv, stratified_checks.csv",
               "оставить tenure_days как непрерывный признак; проверить на валидации, не подбирая порог"),
    }
    plan = pd.DataFrame(HYPOTHESES, columns=["hypothesis_id", "question", "indicator", "groups", "period",
                                             "expected_direction", "limitations"])
    plan["result"] = plan["hypothesis_id"].map(lambda h: results[h][0])
    plan["evidence"] = plan["hypothesis_id"].map(lambda h: f"{results[h][1]}. Источник: {results[h][2]}")
    plan["next_action"] = plan["hypothesis_id"].map(lambda h: results[h][3])
    _w(plan, out / "analysis_plan.csv")

    checks = []
    def chk(name, ok, detail): checks.append(dict(check=name, passed=bool(ok), detail=detail))
    chk("ключ (customer_id, snapshot_at) уникален", not sample.duplicated(["customer_id", "snapshot_at"]).any(),
        f"{len(sample)} строк")
    size = sample.groupby("snapshot_at").size()
    mism = {str(k.date()): (int(size[k]), v) for k, v in indep_all.items() if size.get(k) != v}
    chk("строк на T = независимо рассчитанная активная база", not mism, f"расхождений: {mism or 0}")
    chk("billing_available_at ≤ snapshot_at", (sample["billing_available_at"] <= sample["snapshot_at"]).all(),
        f"нарушений {(sample['billing_available_at'] > sample['snapshot_at']).sum()}")
    chk("source_billing_month = T − 1 месяц",
        (sample["source_billing_month"] == sample["snapshot_at"] - pd.offsets.MonthBegin(1)).all(), "")
    leak_cols = [c for c in sample.columns if "termination" in c or "event_date" in c]
    chk("дата будущего расторжения не в признаках", not leak_cols, f"поля: {leak_cols or 'нет'}")
    chk("метка неполного горизонта неизвестна (не 0); в разбиении таких T нет",
        sample.loc[sample["target_end_exclusive"] > pd.Timestamp(cfg["observation_end"]), "churn_target"].isna().all(),
        f"строк с неизвестной меткой: {int(sample['churn_target'].isna().sum())}")
    be = boundary_example()
    _w(be, tab / "boundary_example.csv")
    be_ok = (be["in_base_at_T"] == be["expected_in_base"]).all() and \
        all((pd.isna(a) and pd.isna(e)) or a == e for a, e in zip(be["churn_target"], be["expected_target"]))
    chk("граничный пример: события ровно на T и на конце горизонта", be_ok, "boundary_example.csv")
    late = avail[avail["report_available_at"] > avail["snapshot_at"]]
    chk("сетевой отчёт после T неизвестен на T", late["failed_probe_rate_at_T"].isna().all(),
        f"отчётов, опубликованных после T: {len(late)} из {len(avail)}")
    total_n, total_c = len(eda), int(eda["churn_target"].sum())
    recon = all(int(t["n_observations"].sum()) == total_n and int(t["n_churn"].sum()) == total_c
                for t in [debt, sup, ten, dyn, by_net, *extra.values()])
    chk("групповые числители и знаменатели восстанавливаются", recon, f"N = {total_n}, уходов {total_c}")
    chk("в EDA не попали валидация/тест", set(eda["snapshot_at"]) == set(eda_dates), "8 дат T")
    checks = pd.DataFrame(checks)
    _w(checks, out / "checks.csv")

    hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(out.rglob("*.csv"))}
    summary = {
        "python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "prepared_dir": "см. config.json (prepared_dir)", "horizon": cfg["horizon"], "base_rule": cfg["base_rule"],
        "splits": cfg["splits"], "eda_rows": total_n, "eda_customers": int(eda["customer_id"].nunique()),
        "eda_churn": total_c, "sample_rows_by_split": sample["split"].value_counts().to_dict(),
        "network_reports_known_at_T": int(avail["report_known_at_T"].sum()),
        "network_reports_unknown_at_T": int((avail["report_known_at_T"] == 0).sum()),
        "checks_passed": bool(checks["passed"].all()), "csv_sha256": hashes,
    }
    (out / "eda_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    say(f"[3] проверки: {'все пройдены' if summary['checks_passed'] else 'ЕСТЬ ОШИБКИ'} ({len(checks)})")
    say(f"Готово: {out}")
    return {"summary": summary, "checks": checks, "plan": plan, "boot": boot, "strat": strat}
