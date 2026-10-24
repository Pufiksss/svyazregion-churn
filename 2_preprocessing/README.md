# Этап 2. Сбор и подготовка данных

Согласованные таблицы абонентов, биллинга, событий договора и обращений в поддержку, месячная таблица `customer_month` и отчёт о качестве. Правила времени и активной базы взяты из устава этапа 1 (`../1_introduction/docs/project_charter.md`).

**Команда.** Алексей — загрузка, очистка, объединение. Захар — проверки качества и доступности, тесты, отчёт, журнал решений, словарь данных, единый запуск.

## Запуск

Из папки `2_preprocessing`, Python ≥ 3.10:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_pipeline.py        
pytest                       
```

Ожидаемый вывод:

```
[1] загружено: subscribers=2600, billing_monthly=24248, contract_events=3299, support_legacy (CSV)=1615, support_current (SQLite)=9373
[3] очистка: support_clean=10818, карантин=40
[4] customer_month: 24098 строк (биллинг 24098)
[5] проверки доступности: OK; ошибок после очистки: 0
```

Пути к файлам и параметры задаются в `config.json`. Чтобы записать результат в другую папку: `python run_pipeline.py --output out2`. Ручных правок файлов нет: повторный запуск даёт те же файлы, SHA-256 сохраняются в `output/reports/quality_summary.json`.

## Входы

| Путь | Что это |
|---|---|
| `data/raw/lesson1/` | subscribers, billing_monthly, contract_events, словарь первого занятия (без изменений) |
| `data/raw/lesson2/support.sqlite` | текущие обращения, читаются SQL-запросом `sql/extract_support.sql` |
| `data/raw/lesson2/support_current_export.csv` | копия SQLite: только сверка, как отдельный источник не используется |
| `data/raw/lesson2/support_legacy_sep_oct_2025.csv` | старая система, 09–10.2025 |
| `data/raw/lesson2/schema_support.csv` | соответствие полей старой и новой системы |
| `data/raw/CHECKSUMS.sha256` | контрольные суммы; при изменении исходника запуск остановится |

## Выходы

| Файл | Содержание |
|---|---|
| `output/subscribers_clean.csv` | 2 600 абонентов, даты подключения восстановлены |
| `output/billing_clean.csv` | 24 098 строк, ключ customer_id + billing_month |
| `output/contract_events_clean.csv` | 3 299 событий |
| `output/support_clean.csv` | 10 818 обращений, время МСК, единые коды тем и каналов |
| `output/quarantine_support.csv` | 40 обращений с причиной исключения |
| `output/customer_month.csv` | 24 098 строк: биллинг + абонент + агрегаты поддержки на дату доступности |
| `output/reports/quality_before.csv`, `quality_after.csv` | проверки до и после очистки |
| `output/reports/availability_checks.csv` | проверки доступности информации |
| `output/reports/decision_log.csv` | журнал решений с числом записей |
| `output/reports/sql_diagnostics.csv` | диагностические SQL-запросы к SQLite |
| `output/reports/quality_summary.json` | машинно-читаемая сводка |

## Документы

- [Отчёт о качестве](docs/quality_report.md)
- [Журнал решений](docs/decision_log.md)
- [Словарь данных v2](docs/data_dictionary_v2.csv)

## Код

| Модуль | Назначение | Ответственный |
|---|---|---|
| `src/prep/load.py` | чтение источников, SQL из SQLite, сверка контрольных сумм | Алексей |
| `src/prep/common.py` | разбор дат и часовых поясов, справочники кодов | Алексей |
| `src/prep/clean.py` | очистка, согласование, карантин, журнал решений | Алексей |
| `src/prep/merge.py` | агрегаты поддержки и customer_month | Алексей |
| `src/prep/pipeline.py`, `run_pipeline.py` | единый запуск | Захар |
| `src/prep/quality.py` | правила проверок до и после | Захар |
| `tests/` | проверки доступности и воспроизводимости | Захар |
