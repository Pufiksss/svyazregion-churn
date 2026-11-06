# Этап 3. Разведочный анализ данных

EDA оттока на подготовленных таблицах этапа 2 (`../2_preprocessing/output`) и новых источниках: сетевых измерениях и календаре событий. Правила базы, горизонта и разбиения — из устава этапа 1.

**Команда.** Алексей — исследовательская таблица снимков, доступность, проверки, запуск. Захар — гипотезы, групповые расчёты, рисунки, отчёт.

## Запуск

Из папки `3_eda`. Этап 2 должен быть уже запущен: нужна папка `../2_preprocessing/output`.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_eda.py                  # таблицы, рисунки, проверки -> output/
pytest                             # 7 тестов, включая повторный запуск
jupyter notebook eda.ipynb         # тот же анализ с пояснениями: Restart kernel -> Run all
```

Параметры (путь к результатам этапа 2, даты T, горизонт, порог стажа, бутстрэп) задаются в `config.json`.

## Правила

| | |
|---|---|
| Единица | «абонент × T», ключ (customer_id, snapshot_at) |
| T | 00:00 МСК 1-го числа; EDA — 01.12.2025…01.07.2026, валидация 01.08.2026, тест 01.09.2026 |
| База | перед событиями на T: `activation < T`, нет termination или `termination ≥ T` |
| Горизонт | `[T, 1-е число следующего месяца)`, конец исключён |
| Признаки | customer_month за T − 1 (available_at = T), справочник, стаж; сеть — только при `report_available_at ≤ T` |

## Результаты

| Файл | Что это |
|---|---|
| `docs/eda_report.md` | отчёт по шаблону |
| `output/analysis_plan.csv` | журнал 4 гипотез с результатами |
| `output/eda_sample.csv` | исследовательская таблица (19 541 строк, из них EDA 15 696) |
| `docs/eda_sample_dictionary.csv` | словарь полей eda_sample |
| `output/network_availability.csv` | доступность сетевых отчётов на (region, measurement_month, snapshot_at) |
| `output/figures/fig1…fig6*.png` | шесть обязательных рисунков |
| `output/tables/*.csv` | расчётные таблицы к рисункам, стратификация, бутстрэп, граничный пример |
| `output/checks.csv`, `output/eda_summary.json` | проверки и машинно-читаемая сводка |
| `data/raw/` | новые источники без изменений + контрольные суммы |
