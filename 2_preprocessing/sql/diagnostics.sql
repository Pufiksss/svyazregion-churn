-- Диагностические запросы к support_current_raw (запускаются пайплайном, результат в output/reports/sql_diagnostics.csv).

-- name: source_rows
SELECT COUNT(*) AS value FROM support_current_raw;

-- name: ticket_ids_with_copies
SELECT COUNT(*) AS value FROM (
  SELECT ticket_id FROM support_current_raw GROUP BY ticket_id HAVING COUNT(*) > 1
);

-- name: full_duplicate_copies
SELECT COUNT(*) - (SELECT COUNT(*) FROM (SELECT DISTINCT * FROM support_current_raw)) AS value
FROM support_current_raw;

-- name: empty_customer_id
SELECT COUNT(*) AS value FROM support_current_raw
WHERE customer_id IS NULL OR TRIM(customer_id) = '';

-- name: customer_id_with_spaces
SELECT COUNT(*) AS value FROM support_current_raw
WHERE customer_id <> TRIM(customer_id);

-- name: created_at_without_timezone
SELECT COUNT(*) AS value FROM support_current_raw WHERE created_at NOT LIKE '%Z';

-- name: empty_topic
SELECT COUNT(*) AS value FROM support_current_raw WHERE topic IS NULL OR TRIM(topic) = '';

-- name: score_out_of_range
SELECT COUNT(*) AS value FROM support_current_raw
WHERE satisfaction_score <> '' AND CAST(satisfaction_score AS INTEGER) NOT BETWEEN 1 AND 5;

-- name: available_before_created_utc
SELECT COUNT(*) AS value FROM support_current_raw
WHERE created_at LIKE '%Z' AND available_at < created_at;
