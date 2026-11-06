# Журнал экспериментов — этап 5

- **Команда:** Алексей, Захар.
- **Версия кода:** папка `5_model`.
- **Версия признаков:** `stage4_v1` — `4_features/output`, 33 признака; SHA-256 файлов в `models/model_card.json`.
- **Периоды:** train — T 01.12.2025–01.07.2026 (8 дат, 15 696 строк, 440 уходов); validation — T 01.08.2026 (1 929 строк, 63 ухода). Test 01.09.2026 не загружался.
- **Основная метрика:** macro Recall@K по датам, K = floor(0,1 × N_active). На validation одна дата, поэтому macro = pooled.
- **Правило равенства:** score ↓, затем customer_id ↑.
- **Правило выбора** (записано в `config.json` до запуска): максимальный validation Recall@K; при равенстве — максимальный AP; дальше — более ранний опыт в списке.

Машинная версия — `output/experiments_journal.csv`, `output/metrics_summary.csv`, `output/metrics_by_date.csv`.

| ID | Метод | Параметры | Seed | Версия X | Fit, с | Статус |
|---|---|---|---:|---|---:|---|
| baseline | Правило | debt_prev_rub ↓, customer_id ↑; неизвестный долг после известных (на данных таких нет) | — | stage4_v1 | — | ok |
| exp01_lr_c01 | LogisticRegression | C = 0.1 | 42 | stage4_v1 | 0,10 | ok, предупреждение (см. ниже) |
| exp02_lr_c1 | LogisticRegression | C = 1.0 | 42 | stage4_v1 | 0,10 | ok, предупреждение |
| exp03_lr_c01_bal | LogisticRegression | C = 0.1, class_weight = balanced | 42 | stage4_v1 | 0,10 | ok, предупреждение |
| exp04_hgb_leaf7 | HistGradientBoosting | max_leaf_nodes 7, max_iter 100, lr 0.07, min_samples_leaf 40, l2 1.0, без early stopping | 42 | stage4_v1 | 0,29 | ok |
| exp05_hgb_leaf15 | HistGradientBoosting | то же, max_leaf_nodes 15 | 42 | stage4_v1 | 0,47 | ok |

Время обучения замерено на облачной машине команды. На другом оборудовании оно будет другим.

| ID | Часть | Recall@K | Precision@K | AP (average_precision_score) | Найдено / ушли | Комментарий |
|---|---|---:|---:|---:|---|---|
| baseline | train | 0,396 | 0,112 | 0,088 | 175 / 440 | планка этапа 4 воспроизведена |
| baseline | validation | 0,222 | 0,073 | 0,058 | 14 / 63 | август тяжелее для правила долга |
| exp01_lr_c01 | train | 0,412 | 0,116 | 0,119 | 181 / 440 | |
| exp01_lr_c01 | validation | 0,238 | 0,078 | 0,061 | 15 / 63 | +1 к baseline |
| exp02_lr_c1 | train | 0,419 | 0,118 | 0,120 | 184 / 440 | |
| exp02_lr_c1 | validation | 0,222 | 0,073 | 0,061 | 14 / 63 | = baseline |
| exp03_lr_c01_bal | train | 0,415 | 0,116 | 0,107 | 182 / 440 | |
| exp03_lr_c01_bal | validation | 0,286 | 0,094 | 0,062 | 18 / 63 | +4 |
| **exp04_hgb_leaf7** | train | 0,543 | 0,153 | 0,307 | 240 / 440 | разрыв 0,23 |
| **exp04_hgb_leaf7** | validation | **0,317** | **0,104** | 0,072 | **20 / 63** | **+6, выбран** |
| exp05_hgb_leaf15 | train | 0,719 | 0,203 | 0,539 | 318 / 440 | разрыв 0,43 — переобучение |
| exp05_hgb_leaf15 | validation | 0,286 | 0,094 | 0,075 | 18 / 63 | |

Таблица по каждой T (N, K, ушли, найдено, Recall@K, Precision@K, AP) — `output/metrics_by_date.csv`.

**Дополнительная временная проверка внутри train** (заявлена заранее, для выбора не используется; `output/inner_temporal_check.csv`). Подготовка и модель обучены заново на T ≤ 01.04.2026, оценка на T 01.05–01.07.2026, 180 уходов, macro Recall@K:

| baseline | exp01 | exp02 | exp03 | exp04 | exp05 |
|---:|---:|---:|---:|---:|---:|
| **0,412** (74) | 0,383 (69) | 0,389 (70) | 0,394 (71) | 0,388 (70) | 0,356 (64) |

**Ошибки и предупреждения.** Ошибок обучения нет, ConvergenceWarning нет. У трёх логистических регрессий записано `OptimizeWarning: Unknown solver options: iprint`. Причина: scikit-learn 1.5.2 передаёт в SciPy 1.18 устаревший параметр `iprint`. Предупреждение относится к логированию решателя и не влияет на результат. Исправление не потребовалось. При обновлении scikit-learn оно исчезнет.

**Выбранный кандидат:** `exp04_hgb_leaf7`. Число дополнительных уходов против baseline на validation: **+6** (20 против 14 из 63). Разложение (`output/validation_overlap_vs_baseline.csv`):
- 13 уходов нашли оба подхода;
- 7 — только модель;
- 1 — только baseline;
- 42 не нашёл никто.

156 из 192 выбранных абонентов совпадают.

**Почему выбран.** Максимальный validation Recall@K по заранее записанному правилу. Решение предварительное, см. отчёт: на внутренней временной проверке baseline не хуже.

**Где сохранён Pipeline и как проверена загрузка.** `models/candidate.joblib`: подготовка и классификатор вместе, списки столбцов и конфигурация. `python check_candidate.py` загружает его в новом процессе с запрещённым `fit` и с помощью стартового `check_saved_model.py` сверяет ключи, score (допуск 1e-12), rank и top-K. Метрики пересчитываются заново: Recall@K 0,3175 (20/63), AP 0,0719 совпадают.
