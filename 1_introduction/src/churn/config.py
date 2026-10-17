"""Параметры проекта, согласованные в уставе (этап 1)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"

CSV_KW = dict(sep=";", encoding="utf-8-sig", dtype=str)

# Дата расчёта — 00:00 МСК первого дня месяца t.
# Горизонт — календарный месяц [t, t + 1 месяц).
HORIZON_MONTHS = 1

# Конец наблюдения: журнал событий полон до этой даты (не включая её).
OBSERVATION_END = "2026-10-01"

# Схема дат расчёта (ретро-срезы)
TRAIN_DATES = [f"2025-{m:02d}-01" for m in (12,)] + [f"2026-{m:02d}-01" for m in range(1, 8)]
VALID_DATE = "2026-08-01"
TEST_DATE = "2026-09-01"        # последний срез с полностью наблюдаемым исходом
SCORING_DATE = "2026-10-01"     # боевой прогноз, исход неизвестен

# Ограничение кампании
CAMPAIGN_SHARE = 0.10

TARIFF_MAP = {"basic": "Basic", "comfort": "Comfort", "premium": "Premium", "премиум": "Premium"}
