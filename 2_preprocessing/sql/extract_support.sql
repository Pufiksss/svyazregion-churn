-- Извлечение текущих обращений из support.sqlite (таблица support_current_raw).
-- Все поля читаются как TEXT без преобразований: очистка выполняется в Python,
-- чтобы аудит качества видел исходные значения.
SELECT ticket_id,
       customer_id,
       created_at,
       available_at,
       topic,
       channel,
       resolved_at,
       resolution_available_at,
       satisfaction_score
FROM support_current_raw
ORDER BY rowid;
