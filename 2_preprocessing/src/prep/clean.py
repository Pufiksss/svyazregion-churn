"""Согласование форматов и очистка. Каждое действие пишется в журнал решений с числом записей."""
from __future__ import annotations

import pandas as pd

from .common import (CHANNEL_RU, TARIFF_MAP, TOPIC_RU, TOPICS, blank, contract_dates, fmt_date,
                     fmt_dt, parse_current, parse_legacy, parse_month, to_num)


class DecisionLog:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, source, field, problem, action, rationale, count, unit):
        self.rows.append(dict(source=source, field=field, problem=problem, action=action,
                              rationale=rationale, affected=int(count), unit=unit))

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# ---------------------------------------------------------------- абоненты
def clean_subscribers(subs: pd.DataFrame, events: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    df = subs.copy()
    for col in df.columns:
        df[col] = df[col].str.strip()
    cd = contract_dates(events)
    act_ev = cd["activation"].reindex(df["customer_id"]).to_numpy()
    own = pd.to_datetime(df["activation_date"].where(df["activation_date"] != ""), format="%Y-%m-%d", errors="coerce")

    missing = own.isna()
    recoverable = missing & ~pd.isna(act_ev)
    df["activation_date_source"] = "subscribers"
    df.loc[recoverable, "activation_date_source"] = "contract_events"
    df.loc[missing & pd.isna(act_ev), "activation_date_source"] = "unrecoverable"
    own = own.where(~recoverable, pd.Series(act_ev, index=df.index))
    df["activation_date"] = fmt_date(own)
    log.add("subscribers", "activation_date", "пропуск даты подключения",
            "восстановлено из contract_events (event_type=activation)",
            "у абонента ровно одно событие activation, оно подтверждает дату однозначно; "
            "у заполненных дат совпадение с журналом 100 %", recoverable.sum(), "ячейки")
    log.add("subscribers", "activation_date", "дату невозможно восстановить",
            "оставить пропуск, activation_date_source=unrecoverable, абонента не удалять",
            "по уставу абонент не исключается из-за даты подключения", (missing & pd.isna(act_ev)).sum(), "ячейки")
    cols = ["customer_id", "activation_date", "activation_date_source", "region",
            "acquisition_channel", "customer_segment"]
    return df[cols].sort_values("customer_id").reset_index(drop=True)


# ---------------------------------------------------------------- события
def clean_events(events: pd.DataFrame, log: DecisionLog) -> pd.DataFrame:
    df = events.copy()
    for col in df.columns:
        df[col] = df[col].str.strip()
    n = len(df)
    df = df.drop_duplicates()
    log.add("contract_events", "*", "полные дубли", "проверено, удалять нечего",
            "журнал полон и корректен по ключу event_id", n - len(df), "повторные копии")
    return df.sort_values(["customer_id", "event_date", "event_type"]).reset_index(drop=True)


# ---------------------------------------------------------------- биллинг
def clean_billing(billing: pd.DataFrame, tariff_price: dict, log: DecisionLog) -> pd.DataFrame:
    n0 = len(billing)
    df = billing.drop_duplicates().copy()
    log.add("billing", "*", "полные дубли строк", "удалены повторные копии, оставлен один экземпляр",
            "технический повтор выгрузки: все поля совпадают", n0 - len(df), "повторные копии")
    df["customer_id"] = df["customer_id"].str.strip()

    month = parse_month(df["billing_month"])
    log.add("billing", "billing_month", "формат DD.MM.YYYY", "приведено к YYYY-MM-01",
            "значение однозначно: первый день месяца в другом формате",
            (~df["billing_month"].str.match(r"^\d{4}-\d{2}-01$")).sum(), "ячейки")
    df["billing_month"] = month

    traw = df["tariff"]
    df["tariff"] = traw.str.strip().str.lower().map(TARIFF_MAP)
    log.add("billing", "tariff", "регистр, пробелы, русское написание (COMFORT, ' basic', Премиум)",
            "приведено к Basic/Comfort/Premium", "monthly_fee_rub совпадает с ценой нормализованного тарифа",
            (traw != df["tariff"]).sum(), "ячейки")

    comma = df["paid_rub"].str.contains(",", regex=False)
    log.add("billing", "paid_rub", "десятичная запятая", "заменена на точку",
            "отличие только в разделителе, сумма не меняется", comma.sum(), "ячейки")
    for col in ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub",
                "traffic_gb", "voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]:
        df[col] = to_num(df[col])

    restore = df["closing_debt_rub"].isna() & df[["opening_debt_rub", "charged_rub", "paid_rub"]].notna().all(axis=1)
    df["closing_debt_restored"] = restore.astype(int)
    df.loc[restore, "closing_debt_rub"] = (df["opening_debt_rub"] + df["charged_rub"] - df["paid_rub"])[restore].round(2)
    log.add("billing", "closing_debt_rub", "пропуск конечного долга",
            "восстановлен по тождеству opening + charged − paid, флаг closing_debt_restored",
            "тождество выполняется на 100 % полных строк; это не статистическая импутация", restore.sum(), "ячейки")

    neg = df["voice_minutes"] < 0
    df["voice_minutes_invalid"] = neg.astype(int)
    df.loc[neg, "voice_minutes"] = pd.NA
    log.add("billing", "voice_minutes", "отрицательные минуты",
            "заменены на пропуск, флаг voice_minutes_invalid; строка и абонент сохранены",
            "истинное значение неизвестно; медиану подбирать только на обучающей выборке позже",
            neg.sum(), "ячейки")

    df["traffic_missing"] = df["traffic_gb"].isna().astype(int)
    log.add("billing", "traffic_gb", "пропуск трафика", "сохранён пропуск, флаг traffic_missing",
            "по словарю ноль и пропуск различаются (нулей в данных нет); глобальная импутация = утечка",
            df["traffic_gb"].isna().sum(), "ячейки")

    df["available_at"] = month + pd.offsets.MonthBegin(1)
    out = df.sort_values(["customer_id", "billing_month"]).reset_index(drop=True)
    conflicts = out.duplicated(["customer_id", "billing_month"]).sum()
    log.add("billing", "customer_id+billing_month", "повтор ключа с разными значениями",
            "проверено после приведения формата месяца", "конфликтов нет, ключ уникален", conflicts, "строки")
    if conflicts:
        raise RuntimeError("Конфликт ключа биллинга требует ручной проверки")
    return out


def format_billing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["billing_month"] = fmt_date(out["billing_month"])
    out["available_at"] = fmt_dt(out["available_at"])
    for col in ["monthly_fee_rub", "charged_rub", "paid_rub", "opening_debt_rub", "closing_debt_rub", "traffic_gb"]:
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else f"{v:.2f}")
    for col in ["voice_minutes", "active_days", "payment_delay_days", "tariff_changes_count"]:
        out[col] = out[col].map(lambda v: "" if pd.isna(v) else str(int(v)))
    return out


# ---------------------------------------------------------------- поддержка
CANON = ["ticket_id", "customer_id", "created_at", "available_at", "topic", "channel",
         "resolved_at", "resolution_available_at", "satisfaction_score"]
DT_COLS = ["created_at", "available_at", "resolved_at", "resolution_available_at"]


def rename_legacy(legacy: pd.DataFrame, schema: pd.DataFrame) -> pd.DataFrame:
    """Переименование полей старой системы по schema_support.csv."""
    mapping = dict(zip(schema["legacy_field"], schema["canonical_field"]))
    return legacy.rename(columns=mapping)[CANON]


def harmonize_support(legacy: pd.DataFrame, current: pd.DataFrame, schema: pd.DataFrame,
                      log: DecisionLog) -> pd.DataFrame:
    """Одна таблица обращений в каноничной схеме, время МСК. Полные копии удаляются по исходной строке."""
    parts = []
    for name, raw in [("legacy_csv", legacy), ("current_sqlite", current)]:
        n0 = len(raw)
        dedup = raw.drop_duplicates()
        log.add(f"support_{name.split('_')[0]}", "*", "полные дубли обращений",
                "удалены повторные копии, разные обращения одного абонента сохранены",
                "совпадают все поля, включая неизменяемый ticket_id", n0 - len(dedup), "повторные копии")
        df = rename_legacy(dedup, schema) if name == "legacy_csv" else dedup[CANON].copy()
        df.insert(0, "source", name)
        df["customer_id_raw"] = df["customer_id"]
        df["customer_id"] = df["customer_id"].str.strip()
        df["ticket_id"] = df["ticket_id"].str.strip()

        if name == "legacy_csv":
            for col in DT_COLS:
                df[col] = parse_legacy(df[col])
            df["created_tz_assumed"] = 0
            topic_raw = df["topic"].str.strip()
            df["topic"] = topic_raw.map(TOPIC_RU)
            df["channel"] = df["channel"].str.strip().map(CHANNEL_RU)
            log.add("support_legacy", "поля", "русские названия полей старой системы",
                    "переименованы по schema_support.csv", "соответствие задано заказчиком", len(df), "строки")
            log.add("support_legacy", "topic, channel", "русские названия тем и каналов",
                    "приведены к кодам network/billing/tariff/services и phone/chat/office",
                    "соответствие из schema_support.csv", len(df), "строки")
            log.add("support_legacy", "даты", "DD.MM.YYYY HH:MM:SS, местное московское время",
                    "разобраны как МСК", "часовой пояс подтверждён паспортом источника", len(df), "строки")
        else:
            naive_mask = None
            for col in DT_COLS:
                df[col], naive = parse_current(df[col])
                if col == "created_at":
                    naive_mask = naive
            df["created_tz_assumed"] = naive_mask.astype(int)
            log.add("support_current", "даты", "UTC с суффиксом Z", "переведены в МСК (+3 ч)",
                    "единый часовой пояс проекта — Москва, момент события сохраняется", len(df), "строки")
            log.add("support_current", "created_at", "время без часового пояса",
                    "принято московским, флаг created_tz_assumed",
                    "подтверждено заказчиком; задержка available_at − created_at при этом в обычном диапазоне 0–36 ч",
                    naive_mask.sum(), "ячейки")
            empty_topic = df["topic"].str.strip() == ""
            df["topic"] = df["topic"].str.strip().where(~empty_topic, "unknown")
            df["channel"] = df["channel"].str.strip()
            log.add("support_current", "topic", "пустая тема",
                    "отдельная категория unknown, считается в support_unknown_topic_count",
                    "обращение было, тема неизвестна; удалять нельзя", empty_topic.sum(), "ячейки")
        parts.append(df)

    sup = pd.concat(parts, ignore_index=True)
    spaces = sup["customer_id_raw"] != sup["customer_id"]
    log.add("support_current", "customer_id", "пробелы вокруг идентификатора", "обрезаны",
            "по схеме пробелы незначимы; после обрезки id найден в справочнике", spaces.sum(), "ячейки")

    score = to_num(sup["satisfaction_score"])
    bad_score = score.notna() & (~score.between(1, 5) | (score % 1 != 0))
    sup["score_invalid"] = bad_score.astype(int)
    sup["satisfaction_score"] = score.where(~bad_score).astype("Int64")
    log.add("support_current", "satisfaction_score", "оценка вне шкалы 1–5 (значение 9)",
            "заменена на пропуск, флаг score_invalid; закрытие сохранено",
            "оценка недостоверна, но факт закрытия корректен", bad_score.sum(), "ячейки")

    bad_res = (sup["resolved_at"] < sup["created_at"]) | (sup["resolution_available_at"] < sup["resolved_at"]) \
        | (sup["resolution_available_at"] < sup["available_at"])
    sup["resolution_invalid"] = bad_res.astype(int)
    for col in ["resolved_at", "resolution_available_at"]:
        sup.loc[bad_res, col] = pd.NaT
    sup.loc[bad_res, "satisfaction_score"] = pd.NA
    log.add("support_current", "resolved_at", "закрытие раньше создания (сдвиг −2 ч)",
            "закрытие и оценка обнулены, флаг resolution_invalid; само обращение учитывается",
            "момент создания и доступности корректны, недостоверна только информация о закрытии",
            bad_res.sum(), "строки")
    return sup


def split_quarantine(sup: pd.DataFrame, subscribers: pd.DataFrame, log: DecisionLog,
                     window: tuple[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Обращения без абонента или без безопасной даты доступности -> карантин с причиной."""
    known = set(subscribers["customer_id"])
    reasons = pd.Series("", index=sup.index)

    def mark(mask, reason):
        nonlocal reasons
        m = mask & (reasons == "")
        reasons = reasons.where(~m, reason)
        return int(m.sum())

    key_conflict = sup.duplicated("ticket_id", keep=False)
    rules = [
        (key_conflict, "ticket_id_conflict", "повтор ticket_id с разными значениями",
         "нельзя выбрать верную версию без заказчика"),
        (sup["customer_id"] == "", "customer_id_missing", "нет customer_id",
         "обращение нельзя связать с абонентом"),
        (~sup["customer_id"].isin(known), "customer_id_unknown", "customer_id нет в справочнике (C999xxx)",
         "новые идентификаторы реальных абонентов не вводятся; связь потеряна"),
        (sup["available_at"] < sup["created_at"], "available_before_created",
         "available_at раньше created_at (сдвиг −5 ч)",
         "нельзя установить безопасный момент доступности; включение в снимок дало бы риск утечки"),
        (sup["created_at"].isna() | sup["available_at"].isna(), "datetime_unparseable",
         "не разбирается дата создания или доступности", "нет безопасной даты"),
        ((sup["created_at"] < window[0]) | (sup["created_at"] >= window[1]), "created_out_of_window",
         "создание вне окна выгрузки", "вне периода источника"),
    ]
    for mask, code, problem, why in rules:
        n = mark(mask, code)
        log.add("support", "*", problem, f"карантин, reason={code}", why, n, "строки")

    sup = sup.assign(billing_month=sup["created_at"].dt.to_period("M").dt.to_timestamp())
    q = sup[reasons != ""].assign(reason=reasons[reasons != ""])
    q["customer_linked"] = q["customer_id"].isin(known).astype(int)
    clean = sup[reasons == ""].copy()
    return clean, q


def format_support(df: pd.DataFrame, quarantine: bool = False) -> pd.DataFrame:
    out = df.copy()
    for col in DT_COLS:
        out[col] = fmt_dt(out[col])
    out["billing_month"] = fmt_date(out["billing_month"])
    out["satisfaction_score"] = out["satisfaction_score"].map(lambda v: "" if pd.isna(v) else str(int(v)))
    cols = ["ticket_id", "customer_id", "source", "created_at", "available_at", "topic", "channel",
            "resolved_at", "resolution_available_at", "satisfaction_score", "billing_month",
            "created_tz_assumed", "score_invalid", "resolution_invalid"]
    if quarantine:
        cols = ["reason", "customer_linked", "customer_id_raw"] + cols
    for c in ["topic", "channel"]:
        out[c] = out[c].fillna("")
    return out[cols].sort_values(["created_at", "ticket_id"]).reset_index(drop=True)
