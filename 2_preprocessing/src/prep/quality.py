"""Проверки качества. Одни и те же правила запускаются до очистки (на исходниках)
и после (на экспортированных *_clean.csv, перечитанных как строки).

Каждая проверка возвращает маску по строкам таблицы и единицу счёта:
  «ячейки» — ошибочные значения одного поля, «строки» — нарушенные правила строки,
  «повторные копии» — лишние экземпляры полностью совпадающих строк.
Пропуски, которые оставлены по решению команды, помечены severity="допустимо".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import (
    CHANNEL_RU,
    CHANNELS,
    CLEAN_DT_RE,
    ISO_NAIVE_RE,
    ISO_Z_RE,
    LEGACY_RE,
    NUM_RE,
    TOPIC_RU,
    TOPICS,
    blank,
    contract_dates,
    parse_clean,
    parse_current,
    parse_legacy,
    parse_month,
    to_num,
)

CELLS, ROWS, COPIES = "ячейки", "строки", "повторные копии"


class Collector:
    def __init__(self, source: str, stage: str, df: pd.DataFrame):
        self.source, self.stage, self.n = source, stage, len(df)
        self.any_issue = pd.Series(False, index=df.index)
        self.items: list[dict] = []

    def add(
        self, field: str, check: str, mask, unit: str, severity: str = "ошибка"
    ):
        mask = (
            pd.Series(mask, index=self.any_issue.index)
            .fillna(False)
            .astype(bool)
        )
        if severity == "ошибка":
            self.any_issue |= mask
        self.items.append(
            dict(
                stage=self.stage,
                source=self.source,
                field=field,
                check=check,
                unit=unit,
                count=int(mask.sum()),
                severity=severity,
            )
        )

    def finish(self) -> list[dict]:
        self.items.insert(
            0,
            dict(
                stage=self.stage,
                source=self.source,
                field="*",
                check="строк в таблице",
                unit=ROWS,
                count=self.n,
                severity="инфо",
            ),
        )
        self.items.append(
            dict(
                stage=self.stage,
                source=self.source,
                field="*",
                check="строк хотя бы с одной ошибкой",
                unit=ROWS,
                count=int(self.any_issue.sum()),
                severity="итог",
            )
        )
        return self.items


def _key_conflicts(df: pd.DataFrame, key: list[str]) -> pd.Series:
    """Строки с повтором ключа после удаления полных копий (= конфликт значений)."""
    dedup = df[~df.duplicated()]
    dup = dedup.duplicated(key, keep=False)
    return dup.reindex(df.index, fill_value=False)


# абоненты
def check_subscribers(
    df: pd.DataFrame, events: pd.DataFrame, stage: str
) -> list[dict]:
    c = Collector("subscribers", stage, df)
    cid = df["customer_id"].str.strip()
    c.add("*", "полные дубли", df.duplicated(), COPIES)
    c.add(
        "customer_id",
        "повтор ключа с разными значениями",
        _key_conflicts(df.assign(customer_id=cid), ["customer_id"]),
        ROWS,
    )
    c.add("activation_date", "пропуск", blank(df["activation_date"]), CELLS)
    fmt_bad = ~blank(df["activation_date"]) & ~df["activation_date"].str.match(
        r"^\d{4}-\d{2}-\d{2}$"
    )
    c.add("activation_date", "неверный формат", fmt_bad, CELLS)
    for col, dom in [
        ("acquisition_channel", {"online", "store", "partner"}),
        ("customer_segment", {"individual", "small_business"}),
    ]:
        c.add(col, "вне справочника значений", ~df[col].isin(dom), CELLS)
    c.add("region", "пропуск", blank(df["region"]), CELLS)
    cd = contract_dates(events)
    act = cd["activation"].reindex(cid).to_numpy()
    sub_act = pd.to_datetime(
        df["activation_date"].where(~blank(df["activation_date"])),
        errors="coerce",
    ).to_numpy()
    c.add(
        "activation_date",
        "не совпадает с событием activation",
        ~pd.isna(sub_act) & ~pd.isna(act) & (sub_act != act),
        ROWS,
    )
    c.add("customer_id", "нет события activation", pd.isna(act), ROWS)
    return c.finish()


# события
def check_events(
    df: pd.DataFrame, subscribers: pd.DataFrame, stage: str
) -> list[dict]:
    c = Collector("contract_events", stage, df)
    cid = df["customer_id"].str.strip()
    c.add("*", "полные дубли", df.duplicated(), COPIES)
    c.add(
        "event_id",
        "повтор ключа с разными значениями",
        _key_conflicts(df, ["event_id"]),
        ROWS,
    )
    c.add(
        "event_type",
        "вне справочника значений",
        ~df["event_type"].isin({"activation", "termination"}),
        CELLS,
    )
    c.add(
        "event_date",
        "неверный формат",
        ~df["event_date"].str.match(r"^\d{4}-\d{2}-\d{2}$"),
        CELLS,
    )
    c.add(
        "customer_id",
        "нет в справочнике абонентов",
        ~cid.isin(set(subscribers["customer_id"].str.strip())),
        ROWS,
    )
    cnt = (
        df.assign(customer_id=cid)
        .groupby(["customer_id", "event_type"])
        .size()
        .unstack(fill_value=0)
    )
    bad_act = cnt.index[cnt.get("activation", 0) != 1]
    bad_ter = cnt.index[cnt.get("termination", 0) > 1]
    c.add("customer_id", "не ровно одно activation", cid.isin(bad_act), ROWS)
    c.add("customer_id", "больше одного termination", cid.isin(bad_ter), ROWS)
    cd = contract_dates(df)
    early = cd.index[cd["termination"] < cd["activation"]]
    c.add(
        "event_date",
        "termination раньше activation",
        cid.isin(early) & (df["event_type"] == "termination"),
        ROWS,
    )
    return c.finish()


# биллинг
NUMERIC = [
    "monthly_fee_rub",
    "charged_rub",
    "paid_rub",
    "opening_debt_rub",
    "closing_debt_rub",
    "traffic_gb",
    "voice_minutes",
    "active_days",
    "payment_delay_days",
    "tariff_changes_count",
]


def check_billing(
    df: pd.DataFrame,
    subscribers: pd.DataFrame,
    events: pd.DataFrame,
    tariff_price: dict,
    stage: str,
) -> list[dict]:
    c = Collector("billing", stage, df)
    cid = df["customer_id"].str.strip()
    month = parse_month(df["billing_month"])
    c.add("*", "полные дубли", df.duplicated(), COPIES)
    c.add(
        "customer_id+billing_month",
        "повтор ключа с разными значениями",
        _key_conflicts(
            df.assign(billing_month=month.astype(str)),
            ["customer_id", "billing_month"],
        ),
        ROWS,
    )
    c.add(
        "billing_month",
        "формат не YYYY-MM-01",
        ~df["billing_month"].str.match(r"^\d{4}-\d{2}-01$"),
        CELLS,
    )
    c.add("billing_month", "не разбирается как дата", month.isna(), CELLS)
    av = pd.to_datetime(
        df["available_at"], format="%Y-%m-%d %H:%M:%S", errors="coerce"
    )
    c.add(
        "available_at",
        "не равно началу следующего месяца",
        av != month + pd.offsets.MonthBegin(1),
        CELLS,
    )
    c.add(
        "tariff",
        "нестандартное написание",
        ~df["tariff"].isin(set(tariff_price)),
        CELLS,
    )

    num = {}
    for col in NUMERIC:
        s = df[col].fillna("").astype(str).str.strip()
        sev = (
            "допустимо"
            if (col in ("traffic_gb", "voice_minutes") and stage == "после")
            else "ошибка"
        )
        c.add(col, "пропуск", s == "", CELLS, severity=sev)
        c.add(
            col,
            "неверный числовой формат",
            (s != "") & ~s.str.match(NUM_RE),
            CELLS,
        )
        num[col] = to_num(s)
        c.add(col, "отрицательное значение", num[col] < 0, CELLS)

    days = month.dt.days_in_month
    c.add(
        "active_days",
        "вне [0, дней в месяце]",
        (num["active_days"] > days),
        CELLS,
    )
    c.add(
        "payment_delay_days",
        "вне [0, дней в месяце]",
        (num["payment_delay_days"] > days),
        CELLS,
    )
    c.add(
        "tariff_changes_count",
        "не 0/1",
        ~num["tariff_changes_count"].isin([0, 1])
        & num["tariff_changes_count"].notna(),
        CELLS,
    )
    tnorm = (
        df["tariff"]
        .str.strip()
        .str.lower()
        .map({k.lower(): k for k in tariff_price} | {"премиум": "Premium"})
    )
    c.add(
        "monthly_fee_rub",
        "не равна цене тарифа",
        num["monthly_fee_rub"] != tnorm.map(tariff_price),
        CELLS,
    )

    diff = (
        num["opening_debt_rub"]
        + num["charged_rub"]
        - num["paid_rub"]
        - num["closing_debt_rub"]
    ).abs()
    c.add(
        "closing_debt_rub",
        "нарушено opening + charged − paid = closing",
        diff > 0.005,
        ROWS,
    )

    # Непрерывность долга: opening текущего месяца = closing предыдущего (по уникальным строкам)
    t = pd.DataFrame(
        {
            "cid": cid,
            "m": month,
            "o": num["opening_debt_rub"],
            "c": num["closing_debt_rub"],
        }
    )
    t = t[~df.duplicated()].sort_values(["cid", "m"])
    prev_c = t.groupby("cid")["c"].shift()
    prev_m = t.groupby("cid")["m"].shift()
    consecutive = prev_m + pd.offsets.MonthBegin(1) == t["m"]
    brk = consecutive & prev_c.notna() & ((t["o"] - prev_c).abs() > 0.005)
    c.add(
        "opening_debt_rub",
        "не равен closing предыдущего месяца",
        brk.reindex(df.index, fill_value=False),
        ROWS,
    )
    first = t.groupby("cid").cumcount() == 0
    c.add(
        "opening_debt_rub",
        "ненулевой в первой строке абонента",
        (first & (t["o"] != 0)).reindex(df.index, fill_value=False),
        ROWS,
    )

    c.add(
        "customer_id",
        "нет в справочнике абонентов",
        ~cid.isin(set(subscribers["customer_id"].str.strip())),
        ROWS,
    )
    cd = contract_dates(events)
    act_m = (
        cd["activation"]
        .reindex(cid)
        .dt.to_period("M")
        .dt.to_timestamp()
        .to_numpy()
    )
    ter_m = (
        cd["termination"]
        .reindex(cid)
        .dt.to_period("M")
        .dt.to_timestamp()
        .to_numpy()
    )
    m = month.to_numpy()
    c.add(
        "billing_month",
        "вне периода договора",
        (m < act_m) | (~pd.isna(ter_m) & (m > ter_m)),
        ROWS,
    )
    return c.finish()


# поддержка
def check_support(
    df: pd.DataFrame,
    subscribers: pd.DataFrame,
    events: pd.DataFrame,
    kind: str,
    stage: str,
    window: tuple[str, str],
    export_at: str,
) -> list[dict]:
    """kind: legacy (поля уже переименованы в каноничные), current, clean."""
    c = Collector(f"support_{kind}", stage, df)
    c.add("*", "полные дубли", df.duplicated(), COPIES)
    c.add(
        "ticket_id",
        "повтор ключа с разными значениями",
        _key_conflicts(df, ["ticket_id"]),
        ROWS,
    )
    c.add(
        "ticket_id",
        "формат не T + 7 цифр",
        ~df["ticket_id"].str.match(r"^T\d{7}$"),
        CELLS,
    )

    raw_cid = df["customer_id"].fillna("")
    cid = raw_cid.str.strip()
    c.add("customer_id", "пропуск", cid == "", CELLS)
    c.add("customer_id", "пробелы вокруг значения", raw_cid != cid, CELLS)
    c.add(
        "customer_id",
        "нет в справочнике абонентов",
        (cid != "") & ~cid.isin(set(subscribers["customer_id"].str.strip())),
        ROWS,
    )

    date_cols = [
        "created_at",
        "available_at",
        "resolved_at",
        "resolution_available_at",
    ]
    parsed = {}
    for col in date_cols:
        s = df[col].fillna("")
        nonblank = ~blank(s)
        if kind == "legacy":
            ok, parsed[col] = s.str.match(LEGACY_RE), parse_legacy(s)
        elif kind == "current":
            ok = s.str.match(ISO_Z_RE)
            parsed[col], naive = parse_current(s)
            c.add(col, "нет часового пояса (принято МСК)", naive, CELLS)
            ok = ok | naive
        else:
            ok, parsed[col] = s.str.match(CLEAN_DT_RE), parse_clean(s)
        c.add(col, "неверный формат даты", nonblank & ~ok, CELLS)
    P = pd.DataFrame(parsed)

    topic, channel = (
        df["topic"].fillna("").str.strip(),
        df["channel"].fillna("").str.strip(),
    )
    topic_dom = (
        set(TOPIC_RU)
        if kind == "legacy"
        else TOPICS | ({"unknown"} if kind == "clean" else set())
    )
    chan_dom = set(CHANNEL_RU) if kind == "legacy" else CHANNELS
    c.add("topic", "пропуск", topic == "", CELLS)
    c.add(
        "topic",
        "вне справочника значений",
        (topic != "") & ~topic.isin(topic_dom),
        CELLS,
    )
    if kind == "clean":
        c.add(
            "topic",
            "unknown (пропуск темы сохранён категорией)",
            topic == "unknown",
            CELLS,
            severity="допустимо",
        )
    c.add("channel", "вне справочника значений", ~channel.isin(chan_dom), CELLS)

    score = to_num(df["satisfaction_score"])
    c.add(
        "satisfaction_score",
        "вне 1–5 или не целое",
        score.notna() & (~score.between(1, 5) | (score % 1 != 0)),
        CELLS,
    )
    c.add(
        "satisfaction_score",
        "оценка без закрытия",
        score.notna() & P["resolution_available_at"].isna(),
        ROWS,
    )
    c.add(
        "satisfaction_score",
        "пропуск (допустим)",
        score.isna(),
        CELLS,
        severity="допустимо",
    )

    c.add(
        "available_at",
        "раньше created_at",
        P["available_at"] < P["created_at"],
        ROWS,
    )
    c.add(
        "resolved_at",
        "раньше created_at",
        P["resolved_at"] < P["created_at"],
        ROWS,
    )
    c.add(
        "resolution_available_at",
        "раньше resolved_at",
        P["resolution_available_at"] < P["resolved_at"],
        ROWS,
    )
    c.add(
        "resolution_available_at",
        "раньше available_at",
        P["resolution_available_at"] < P["available_at"],
        ROWS,
    )
    c.add(
        "resolved_at",
        "закрытие без даты доступности или наоборот",
        P["resolved_at"].isna() != P["resolution_available_at"].isna(),
        ROWS,
    )
    c.add(
        "created_at",
        "вне окна выгрузки",
        (P["created_at"] < window[0]) | (P["created_at"] >= window[1]),
        ROWS,
    )
    late = (P["available_at"] > export_at) | (
        P["resolution_available_at"] > export_at
    )
    c.add("available_at", "позже формирования выгрузки", late, ROWS)

    cd = contract_dates(events)
    act = cd["activation"].reindex(cid).to_numpy()
    ter = cd["termination"].reindex(cid).to_numpy()
    cr = P["created_at"].to_numpy()
    c.add(
        "created_at",
        "до подключения абонента",
        ~pd.isna(act) & (cr < act),
        ROWS,
    )
    c.add(
        "created_at",
        "после расторжения договора",
        ~pd.isna(ter) & (cr >= ter),
        ROWS,
    )
    return c.finish()
