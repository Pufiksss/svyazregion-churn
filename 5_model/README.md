# Этап 5. Построение и обучение модели

Обучение двух методов (логистическая регрессия, HistGradientBoosting) на признаках этапа 4 (`../4_features/output`), сравнение с правилом долга, выбор кандидата по validation и сохранение Pipeline. Test 01.09.2026 не используется.

**Команда.** Алексей — обучающий цикл, сохранение, проверка загрузки, тесты. Захар — план конфигураций, правило выбора, анализ результатов, журнал и отчёт.

## Запуск

Из папки `5_model`. Этап 4 должен быть уже запущен.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python run_training.py             # baseline + 5 опытов, выбор, models/candidate.joblib
python check_candidate.py          # новый процесс: загрузка без fit, сверка score / top-K / метрик
pytest                             # 7 тестов
```

Признаки, опыты, правило выбора, K и seed задаются в `config.json`.

## Результат

| | Validation Recall@K | Найдено / ушли |
|---|---:|---|
| Baseline (долг) | 0,222 | 14 / 63 |
| Выбран: exp04_hgb_leaf7 | 0,317 | 20 / 63 |

На внутренней временной проверке train baseline не хуже, поэтому преимущество модели не доказано. Подробности — в [отчёте](docs/training_report.md) и [журнале](docs/experiments_journal.md).

## Файлы

| Путь | Что это |
|---|---|
| `src/train.py` | обучающий цикл |
| `src/training_utils.py`, `model_factory.py`, `preprocessing_template.py`, `check_saved_model.py` | стартовые модули курса (без изменений) |
| `models/candidate.joblib`, `models/model_card.json` | Pipeline целиком и его паспорт |
| `output/experiments_journal.csv`, `metrics_summary.csv`, `metrics_by_date.csv` | журнал и метрики train/validation по T |
| `output/inner_temporal_check.csv` | дополнительная временная проверка внутри train |
| `output/validation_predictions.csv` | ключи, score, rank, top-K выбранного кандидата |
| `output/validation_labels_for_check.csv` | метки validation отдельно, для пересчёта метрик |
| `output/validation_overlap_vs_baseline.csv` | пересечение найденных уходов с baseline |
