-- 02_zscore_outliers.sql
-- Z-Score Outlier Detection: Classic statistical method
-- Technique: (value - mean) / stddev > 3
-- Pattern: CTE for stats + CROSS JOIN to apply thresholds

WITH stats AS (
    SELECT
        AVG(cost_usd)        AS avg_cost,
        STDDEV(cost_usd)     AS std_cost,
        AVG(accuracy)        AS avg_acc,
        STDDEV(accuracy)     AS std_acc,
        AVG(training_hours)  AS avg_hours,
        STDDEV(training_hours) AS std_hours,
        AVG(dataset_size)    AS avg_ds,
        STDDEV(dataset_size) AS std_ds
    FROM mlops_demo_db.ml_training_runs
)
SELECT
    r.run_id,
    r.model_name,
    r.cost_usd,
    r.accuracy,
    r.training_hours,
    r.dataset_size,
    ROUND(ABS(r.cost_usd - s.avg_cost) / NULLIF(s.std_cost, 0), 2) AS cost_zscore,
    ROUND(ABS(r.accuracy - s.avg_acc) / NULLIF(s.std_acc, 0), 2) AS acc_zscore,
    ROUND(ABS(r.training_hours - s.avg_hours) / NULLIF(s.std_hours, 0), 2) AS hours_zscore,
    ROUND(ABS(CAST(r.dataset_size AS DOUBLE) - s.avg_ds) / NULLIF(s.std_ds, 0), 2) AS ds_zscore
FROM mlops_demo_db.ml_training_runs r
CROSS JOIN stats s
WHERE ABS(r.cost_usd - s.avg_cost) / NULLIF(s.std_cost, 0) > 3
   OR ABS(r.accuracy - s.avg_acc) / NULLIF(s.std_acc, 0) > 3
   OR ABS(r.training_hours - s.avg_hours) / NULLIF(s.std_hours, 0) > 3
   OR ABS(CAST(r.dataset_size AS DOUBLE) - s.avg_ds) / NULLIF(s.std_ds, 0) > 3
ORDER BY cost_zscore DESC;
