# Этап 4. Формирование признаков и выбор метода

Признаки на даты расчёта T из очищенных таблиц этапа 2 (`../2_preprocessing/output`) и нового источника `digital_usage_monthly.csv`, преобразователь с fit только на train, baseline по долгу и план моделирования. Правила базы, горизонта и разбиения — из устава этапа 1, выбор признаков — по выводам EDA этапа 3.

**Команда.** Алексей — снимки и признаки, слои данных, проверки времени. Захар — паспорт признаков, преобразователь, baseline, план моделирования.

## Запуск

Из папки `4_features`. Этап 2 должен быть уже запущен.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_features.py
pytest                             # 9 тестов, включая повторный запуск
```

Даты, окно, горизонт, режим активной базы и доля кампании задаются в `config.json`.

## Результаты (`output/`)

| Файл | Что это |
|---|---|
| `features.csv` | ключи + 33 признака, пропуски сохранены (19 541 строка) |
| `labels.csv` | ключи, `churn_target`, `target_end_exclusive` |
| `split.csv` | ключи и train / validation / test |
| `feature_audit.csv` | окна, максимальные моменты доступности использованных строк, полнота истории, статус цифрового отчёта |
| `feature_catalog.csv` | паспорт признаков (27 содержательных + 6 полноты) |
| `x_columns.csv`, `x_shapes.csv`, `train_medians.csv` | столбцы X после преобразования, размеры матриц, статистики train |
| `baseline_train.csv` | Recall@K и Precision@K правила долга по датам train с числителями и знаменателями |
| `checks.csv`, `features_summary.json` | проверки и сводка |

План моделирования: [docs/modeling_plan.md](docs/modeling_plan.md).

Все слои упорядочены одинаково по (snapshot_at, customer_id). Строки матрицы X идут в том же порядке, что строки `features.csv` нужной части.
