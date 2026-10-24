"""Единый запуск подготовки: исходники -> чистые таблицы, customer_month, карантин, отчёты."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import clean, merge, quality
from .common import parse_clean, parse_month
from .load import Config, load_all, read_csv, run_diagnostics, verify_checksums


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(
        path, sep=";", index=False, encoding="utf-8-sig", lineterminator="\n"
    )


def _support_typed(path: Path) -> pd.DataFrame:
    """Перечитывает support_clean/quarantine и восстанавливает типы для агрегации."""
    df = read_csv(path)
    for col in clean.DT_COLS:
        df[col] = parse_clean(df[col])
    df["billing_month"] = parse_month(df["billing_month"])
    df["satisfaction_score"] = pd.to_numeric(
        df["satisfaction_score"].replace("", None)
    ).astype("Int64")
    for col in ["created_tz_assumed", "score_invalid", "resolution_invalid"] + (
        ["customer_linked"] if "customer_linked" in df else []
    ):
        df[col] = df[col].astype(int)
    return df


def availability_checks(
    cm: pd.DataFrame, support: pd.DataFrame
) -> pd.DataFrame:
    """Шаг 5: независимый пересчёт правил доступности на реальных данных."""
    rows = []
    s = support.assign(
        snapshot=support["billing_month"] + pd.offsets.MonthBegin(1)
    )
    created_month = s["created_at"].dt.to_period("M").dt.to_timestamp()
    rows.append(
        (
            "месяц создания = billing_month (МСК)",
            "0 расхождений",
            int((created_month != s["billing_month"]).sum()),
        )
    )

    eligible = s[s["available_at"] <= s["snapshot"]]
    recount = eligible.groupby(["customer_id", "billing_month"]).size()
    got = cm.set_index(["customer_id", "billing_month"])["support_count"]
    diff = got.sub(recount.reindex(got.index, fill_value=0)).abs().sum()
    rows.append(
        (
            "support_count = пересчёт по available_at ≤ snapshot",
            "0 расхождений",
            int(diff),
        )
    )

    late = s[(s["available_at"] > s["snapshot"])]
    rows.append(
        (
            "обращения месяца, поступившие после снимка (не входят)",
            "исключены",
            len(late),
        )
    )
    boundary = late[
        late["created_at"] >= late["snapshot"] - pd.Timedelta(hours=1)
    ]
    rows.append(
        (
            "граничные: созданы в последний час месяца, загружены после 00:00",
            "исключены",
            len(boundary),
        )
    )

    res_ok = eligible["resolution_available_at"].notna() & (
        eligible["resolution_available_at"] <= eligible["snapshot"]
    )
    rec_res = res_ok.groupby(
        [eligible["customer_id"], eligible["billing_month"]]
    ).sum()
    diff_res = (
        cm.set_index(["customer_id", "billing_month"])["support_resolved_count"]
        .sub(rec_res.reindex(got.index, fill_value=0))
        .abs()
        .sum()
    )
    rows.append(
        (
            "support_resolved_count учитывает только resolution_available_at ≤ snapshot",
            "0 расхождений",
            int(diff_res),
        )
    )
    closed_later = int(
        (eligible["resolution_available_at"] > eligible["snapshot"]).sum()
    )
    rows.append(
        (
            "закрытия, ставшие известны после снимка (не учтены)",
            "исключены",
            closed_later,
        )
    )

    future_cols = [
        c
        for c in cm.columns
        if any(w in c.lower() for w in ("termination", "churn", "target"))
    ]
    rows.append(
        (
            "будущие termination / целевая переменная в customer_month",
            "0 полей",
            len(future_cols),
        )
    )
    snap_ok = (
        pd.to_datetime(cm["available_at"])
        == cm["billing_month"] + pd.offsets.MonthBegin(1)
    ).all()
    rows.append(
        (
            "available_at строки = 1-е число следующего месяца",
            "все строки",
            int(snap_ok),
        )
    )
    df = pd.DataFrame(rows, columns=["check", "expected", "result"])
    df["passed"] = [r == 0 if e.startswith("0") else True for _, e, r in rows]
    df.loc[df["check"].str.startswith("available_at строки"), "passed"] = bool(
        snap_ok
    )
    return df


def run(
    config_path: Path | None = None,
    output_dir: Path | None = None,
    verbose: bool = True,
) -> dict:
    cfg = Config.load(config_path, output_dir)
    out, rep = cfg.out, cfg.out / "reports"
    rep.mkdir(parents=True, exist_ok=True)
    C = cfg.raw
    window, export_at, price = (
        tuple(C["support_window"]),
        C["final_export_at"],
        C["tariff_price"],
    )
    say = print if verbose else (lambda *a, **k: None)

    # 1. загрузка
    checksums = verify_checksums(cfg)
    src = load_all(cfg)
    diag = run_diagnostics(
        cfg.raw_path("lesson2", "sqlite"), cfg.path("sql_diagnostics")
    )
    _write(diag, rep / "sql_diagnostics.csv")
    subs, bill, ev = src["subscribers"], src["billing"], src["events"]
    legacy_canon = clean.rename_legacy(
        src["support_legacy"], src["schema_support"]
    )
    inventory = []
    for name, df, unit, key in [
        ("subscribers", subs, "абонент", "customer_id"),
        (
            "billing_monthly",
            bill,
            "абонент × месяц",
            "customer_id + billing_month",
        ),
        ("contract_events", ev, "событие договора", "event_id"),
        (
            "support_legacy (CSV)",
            src["support_legacy"],
            "обращение",
            "ticket_no → ticket_id",
        ),
        (
            "support_current (SQLite)",
            src["support_current"],
            "обращение",
            "ticket_id",
        ),
    ]:
        inventory.append(
            dict(
                source=name,
                rows=len(df),
                unit=unit,
                key=key,
                columns=len(df.columns),
            )
        )
    say(
        "[1] загружено: "
        + ", ".join(f"{i['source']}={i['rows']}" for i in inventory)
    )

    # 2. проверки до
    before = []
    before += quality.check_subscribers(subs, ev, "до")
    before += quality.check_events(ev, subs, "до")
    before += quality.check_billing(bill, subs, ev, price, "до")
    before += quality.check_support(
        legacy_canon, subs, ev, "legacy", "до", window, export_at
    )
    before += quality.check_support(
        src["support_current"], subs, ev, "current", "до", window, export_at
    )
    before = pd.DataFrame(before)
    _write(before, rep / "quality_before.csv")

    # 3. очистка
    log = clean.DecisionLog()
    subs_c = clean.clean_subscribers(subs, ev, log)
    ev_c = clean.clean_events(ev, log)
    bill_c = clean.clean_billing(bill, price, log)
    sup_all = clean.harmonize_support(
        src["support_legacy"],
        src["support_current"],
        src["schema_support"],
        log,
    )
    sup_c, quar = clean.split_quarantine(sup_all, subs_c, log, window)

    _write(subs_c, out / "subscribers_clean.csv")
    _write(ev_c, out / "contract_events_clean.csv")
    _write(clean.format_billing(bill_c), out / "billing_clean.csv")
    _write(clean.format_support(sup_c), out / "support_clean.csv")
    _write(
        clean.format_support(quar, quarantine=True),
        out / "quarantine_support.csv",
    )
    say(f"[3] очистка: support_clean={len(sup_c)}, карантин={len(quar)}")

    # 4. объединение (агрегация по перечитанным файлам -> проверяется именно экспорт)
    sup_t = _support_typed(out / "support_clean.csv")
    quar_t = _support_typed(out / "quarantine_support.csv")
    cm, control = merge.build_customer_month(bill_c, subs_c, sup_t, quar_t)
    cm_out = clean.format_billing(cm)
    cm_out["support_score_mean"] = cm["support_score_mean"].map(
        lambda v: "" if pd.isna(v) else f"{v:.3f}"
    )
    _write(cm_out, out / "customer_month.csv")
    say(
        f"[4] customer_month: {len(cm)} строк (биллинг {control['billing_rows']})"
    )

    # 5. доступность
    avail = availability_checks(cm, sup_t)
    _write(avail, rep / "availability_checks.csv")

    # 2'. проверки после (по экспортированным файлам)
    subs_r, ev_r = (
        read_csv(out / "subscribers_clean.csv"),
        read_csv(out / "contract_events_clean.csv"),
    )
    bill_r, sup_r = (
        read_csv(out / "billing_clean.csv"),
        read_csv(out / "support_clean.csv"),
    )
    after = []
    after += quality.check_subscribers(subs_r, ev_r, "после")
    after += quality.check_events(ev_r, subs_r, "после")
    after += quality.check_billing(bill_r, subs_r, ev_r, price, "после")
    after += quality.check_support(
        sup_r, subs_r, ev_r, "clean", "после", window, export_at
    )
    after = pd.DataFrame(after)
    _write(after, rep / "quality_after.csv")

    decisions = log.frame()
    _write(decisions, rep / "decision_log.csv")

    # сводка
    err_after = after[(after["severity"] == "ошибка")]["count"].sum()
    summary = {
        "python": platform.python_version(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "raw_checksums_ok": all(checksums.values()),
        "sqlite_equals_csv_backup": src["_sqlite_equals_csv_backup"],
        "load_method_current": "SQLite (sql/extract_support.sql), CSV-копия только для сверки",
        "inventory": inventory,
        "rows_out": {
            "subscribers_clean": len(subs_c),
            "billing_clean": len(bill_c),
            "contract_events_clean": len(ev_c),
            "support_clean": len(sup_c),
            "quarantine_support": len(quar),
            "customer_month": len(cm),
        },
        "quarantine_by_reason": quar["reason"].value_counts().to_dict(),
        "quarantine_customer_linked": int(quar["customer_linked"].sum()),
        "rows_with_errors_before": before[
            before["check"] == "строк хотя бы с одной ошибкой"
        ]
        .set_index("source")["count"]
        .to_dict(),
        "rows_with_errors_after": after[
            after["check"] == "строк хотя бы с одной ошибкой"
        ]
        .set_index("source")["count"]
        .to_dict(),
        "error_cells_or_rows_after": int(err_after),
        "merge_control": control,
        "availability_checks_passed": bool(avail["passed"].all()),
        "subscribers_activation_source": subs_c["activation_date_source"]
        .value_counts()
        .to_dict(),
    }
    hashes = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(out.glob("*.csv"))
    }
    summary["output_sha256"] = hashes
    (rep / "quality_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    say(
        f"[5] проверки доступности: {'OK' if summary['availability_checks_passed'] else 'ОШИБКА'}; "
        f"ошибок после очистки: {err_after}"
    )
    say(f"Готово: {out}")
    return summary
