from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "axes.grid.axis": "y", "axes.axisbelow": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "figure.dpi": 110,
})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _pct(ax):
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))


def fig1_missing(miss: pd.DataFrame, period: str, path):
    m = miss.sort_values("missing_share")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(m["field"], m["missing_share"], color=BLUE, height=0.6)
    for y, (s, n) in enumerate(zip(m["missing_share"], m["n_missing"])):
        ax.text(s, y, f"  {n} ({s:.2%})", va="center", color=INK2, fontsize=9)
    ax.grid(True, axis="x"); ax.grid(False, axis="y")
    ax.set_xlim(0, max(m["missing_share"].max() * 1.6, 0.02))
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=1))
    ax.set_xlabel("доля пропусков, % строк «абонент × T»")
    ax.set_title(f"Рис. 1. Пропуски исследуемых показателей\n{period}, N = {int(m['n_rows'].iloc[0])} строк")
    fig.text(0.01, 0.01, "traffic_gb — пропуск в биллинге; voice_minutes — отрицательные минуты заменены пропуском. "
             "Остальные поля полные.", fontsize=8, color=INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig2_distributions(df: pd.DataFrame, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.8))
    t = df["traffic_gb"].dropna()
    a.hist(t, bins=np.arange(0, t.quantile(0.995) + 5, 5), color=BLUE, edgecolor=SURFACE, linewidth=1)
    a.axvline(t.median(), color=INK, lw=1.5, ls="--")
    a.text(t.median(), a.get_ylim()[1] * 0.92, f" медиана {t.median():.1f} ГБ", color=INK, fontsize=9)
    a.set_xlabel("трафик предыдущего месяца, ГБ (обрезано по p99.5)")
    a.set_ylabel("строк")
    a.set_title(f"Трафик: n = {len(t)}, пропусков {df['traffic_gb'].isna().sum()}\n"
                f"p90 = {t.quantile(.9):.1f}, max = {t.max():.0f} ГБ", fontsize=10)

    d = df["closing_debt_rub"].dropna()
    pos = d[d > 0]
    b.hist(pos, bins=np.arange(0, pos.quantile(0.99) + 100, 100), color=ORANGE, edgecolor=SURFACE, linewidth=1)
    b.axvline(pos.median(), color=INK, lw=1.5, ls="--")
    b.text(pos.median(), b.get_ylim()[1] * 0.92, f" медиана {pos.median():.0f} ₽", color=INK, fontsize=9)
    b.set_xlabel("долг на конец предыдущего месяца, ₽ (только > 0, обрезано по p99)")
    b.set_title(f"Долг: n = {len(d)}, нулевой у {(d == 0).mean():.1%}\nсреди >0: n = {len(pos)}, p90 = {pos.quantile(.9):.0f}, "
                f"max = {pos.max():.0f} ₽", fontsize=10)
    fig.suptitle("Рис. 2. Распределения трафика и долга (EDA-период, строки «абонент × T»)", fontweight="bold")
    _save(fig, path)


def fig3_dynamics(t: pd.DataFrame, path):
    x = pd.to_datetime(t["snapshot_at"]).dt.strftime("%m.%Y")
    fig, (a, b) = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True)
    a.plot(x, t["churn_rate"], color=BLUE, lw=2, marker="o", ms=7)
    for i, r in t.reset_index(drop=True).iterrows():
        a.annotate(f"{r['n_churn']}/{r['n_observations']}", (i, r["churn_rate"]), textcoords="offset points",
                   xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    _pct(a)
    a.set_ylim(0, t["churn_rate"].max() * 1.3)
    a.set_ylabel("доля расторжений в [T, T+1 мес.)")
    a.set_title("Рис. 3. Доля оттока и размер активной базы по датам расчёта T (EDA-период)")
    b.plot(x, t["n_observations"], color=AQUA, lw=2, marker="o", ms=7)
    for i, n in enumerate(t["n_observations"]):
        b.annotate(f"{n}", (i, n), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    b.set_ylim(t["n_observations"].min() - 30, t["n_observations"].max() + 30)
    b.set_ylabel("активная база на T, абонентов\n(ось не от нуля)")
    b.set_xlabel("дата расчёта T")
    _save(fig, path)


def _bars(ax, t, col, order, color, title):
    t = t.set_index(col).reindex([o for o in order if o in set(t[col])]).reset_index()
    ax.bar(t[col].astype(str), t["churn_rate"], color=color, width=0.55)
    for i, r in t.iterrows():
        flag = " ⚠ <30" if r["small_group"] else ""
        ax.text(i, r["churn_rate"], f"{r['churn_rate']:.2%}\n{r['n_churn']}/{r['n_observations']}{flag}",
                ha="center", va="bottom", fontsize=8, color=INK2)
    _pct(ax)
    ax.set_ylim(0, t["churn_rate"].max() * 1.35)
    ax.set_title(title, fontsize=10)


def fig4_debt_support(debt: pd.DataFrame, sup: pd.DataFrame, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    _bars(a, debt, "debt_group", ["долг = 0", "долг > 0", "неизвестно"], ORANGE, "по долгу на конец прошлого месяца")
    a.set_ylabel("доля оттока (уходы / строки)")
    _bars(b, sup, "support_group", ["0", "1", "2+", "нет данных"], BLUE, "по числу доступных обращений за прошлый месяц")
    fig.suptitle("Рис. 4. Доля оттока по долгу и обращениям в поддержку (EDA-период; подписи: уходы/строки)",
                 fontweight="bold")
    _save(fig, path)


def fig5_segment(ten: pd.DataFrame, ten_by_debt: pd.DataFrame, path):
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.4))
    _bars(a, ten, "tenure_group", ["<90 дн.", "≥90 дн."], AQUA, "по стажу на T")
    a.set_ylabel("доля оттока (уходы / строки)")
    groups = ["долг = 0", "долг > 0"]
    w = 0.38
    for j, (g, c) in enumerate(zip(groups, [BLUE, ORANGE])):
        sub = ten_by_debt[ten_by_debt["debt_group"] == g].set_index("tenure_group").reindex(["<90 дн.", "≥90 дн."])
        xs = np.arange(2) + (j - 0.5) * w
        b.bar(xs, sub["churn_rate"], width=w, color=c, label=g)
        for x, (_, r) in zip(xs, sub.iterrows()):
            b.text(x, r["churn_rate"], f"{r['churn_rate']:.1%}\n{int(r['n_churn'])}/{int(r['n_observations'])}",
                   ha="center", va="bottom", fontsize=7.5, color=INK2)
    b.set_xticks(range(2), ["<90 дн.", "≥90 дн."])
    _pct(b)
    b.set_ylim(0, ten_by_debt["churn_rate"].max() * 1.4)
    b.legend(frameon=False, loc="upper left")
    b.set_title("стаж × долг: различие стажа внутри групп долга", fontsize=10)
    fig.suptitle("Рис. 5. Доля оттока по стажу (<90 / ≥90 дней) и проверка состава групп", fontweight="bold")
    _save(fig, path)


def fig6_network(nm: pd.DataFrame, pooled: pd.DataFrame, eda_months, path):
    regions = sorted(nm["region"].unique())
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.6), sharey=True)
    ymax = nm["failed_probe_rate"].max() * 1.25
    for ax, reg in zip(axes.flat, regions + ["__pooled__"]):
        if reg == "__pooled__":
            p = pooled[pooled["measurement_month"].isin(eda_months)]
            x = p["measurement_month"].dt.strftime("%m.%y")
            ax.plot(x, p["failed_probe_rate_pooled"], color=INK2, lw=2, marker="o", ms=6)
            ax.set_title("все регионы: Σ ошибок / Σ проб", fontsize=10)
        else:
            d = nm[(nm["region"] == reg) & nm["measurement_month"].isin(eda_months)]
            x = d["measurement_month"].dt.strftime("%m.%y")
            ax.plot(x, d["failed_probe_rate"], color=BLUE, lw=2, zorder=1)
            k = d["report_known_at_T"] == 1
            ax.scatter(x[k], d["failed_probe_rate"][k], s=48, color=BLUE, zorder=2, label="отчёт известен на T")
            ax.scatter(x[~k], d["failed_probe_rate"][~k], s=48, facecolor=SURFACE, edgecolor=ORANGE, lw=2,
                       zorder=3, label="отчёт опубликован после T")
            for xi, (_, r) in zip(x, d.iterrows()):
                ax.annotate(f"{r['probe_count']}", (xi, r["failed_probe_rate"]), textcoords="offset points",
                            xytext=(0, -13), ha="center", fontsize=6.5, color=INK2)
            ax.set_title(reg, fontsize=10)
        _pct(ax)
        ax.set_ylim(0, ymax)
        ax.tick_params(axis="x", labelrotation=45, labelsize=8)
    axes.flat[0].set_ylabel("доля неуспешных проб")
    axes.flat[3].set_ylabel("доля неуспешных проб")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False)
    fig.suptitle("Рис. 6. Неуспешные технические пробы по регионам, месяцы измерений 11.2025–06.2026 "
                 "(T = месяц + 1; подписи — число проб)", fontweight="bold")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)
