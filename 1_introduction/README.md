# Прогноз оттока абонентов СвязьРегион

Учебный проект: ежемесячно выбирать до 10 % активной базы с наибольшим риском расторжения договора в следующем календарном месяце.

**Команда:** Алексей — руководитель проекта, данные и ML; Захар — бизнес-аналитик, метрики, коммуникация с заказчиком. Подробно — в [уставе](docs/project_charter.md#1-команда-и-роли).

## Постановка коротко

- **Объект:** пара (абонент, дата расчёта t).
- **Дата расчёта:** 00:00 МСК 1-го числа месяца; признаки — биллинг с `available_at ≤ t` и постоянные атрибуты.
- **Активная база:** `activation < t` и (нет `termination` или `termination ≥ t`).
- **Отток:** событие `termination` в `contract_events.csv` с датой в `[t, t + 1 мес.)`.
- **Выход:** скор, ранг и флаг `in_campaign` (не более ⌊0,1 × база⌋ абонентов).
- **Метрика:** Recall@10 %, дополнительно Precision@10 %, Lift@10 %, PR-AUC.

## Документы этапа 1

| Файл | Содержание | Ответственный |
|---|---|---|
| [docs/project_charter.md](docs/project_charter.md) | Устав проекта | Захар |
| [docs/questions_and_assumptions.md](docs/questions_and_assumptions.md) | Вопросы заказчику, допущения, требования к обращениям | Захар |
| [docs/data_audit.md](docs/data_audit.md) | Ключи, проблемы качества, база и отток по срезам | Алексей |
| [notebooks/01_data_audit.ipynb](notebooks/01_data_audit.ipynb) | Воспроизводимый аудит | Алексей |
| [src/churn/target.py](src/churn/target.py) | Активная база, целевая переменная, срез по доступности | Алексей |

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest                      # проверки ключей, базы, горизонта, утечки
jupyter notebook notebooks/01_data_audit.ipynb
```

Данные в `data/raw/` — исходные выгрузки (UTF-8 с BOM, разделитель `;`), только чтение.
