-- 03_iqr_outliers.sql
-- IQR (Interquartile Range) Outlier Detection: Tukey's fences
-- Technique: value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
-- More robust than Z-score for skewed distributions

WITH quartiles AS (
    SELECT
        APPROX_PERCENTILE(cost_usd, 0.25) AS cost_q1,
        APPROX_PERCENTILE(cost_usd, 0.75) AS cost_q3,
        APPROX_PERCENTILE(training_hours, 0.25) AS hours_q1,
        APPROX_PERCENTILE(training_hours, 0.75) AS hours_q3,
        APPROX_PERCENTILE(dataset_size, 0.25) AS ds_q1,
        APPROX_PERCENTILE(dataset_size, 0.75) AS ds_q3
    FROM mlops_demo_db.ml_training_runs
),
bounds AS (
    SELECT
        cost_q1 - 1.5 * (cost_q3 - cost_q1) AS cost_lower,
        cost_q3 + 1.5 * (cost_q3 - cost_q1) AS cost_upper,
        hours_q1 - 1.5 * (hours_q3 - hours_q1) AS hours_lower,
        hours_q3 + 1.5 * (hours_q3 - hours_q1) AS hours_upper,
        ds_q1 - 1.5 * (ds_q3 - ds_q1) AS ds_lower,
        ds_q3 + 1.5 * (ds_q3 - ds_q1) AS ds_upper
    FROM quartiles
)
SELECT
    r.run_id,
    r.model_name,
    r.cost_usd,
    r.training_hours,
    r.dataset_size,
    CASE WHEN r.cost_usd < b.cost_lower OR r.cost_usd > b.cost_upper
         THEN 'YES' ELSE 'no' END AS cost_outlier,
    CASE WHEN r.training_hours < b.hours_lower OR r.training_hours > b.hours_upper
         THEN 'YES' ELSE 'no' END AS hours_outlier,
    CASE WHEN r.dataset_size < b.ds_lower OR r.dataset_size > b.ds_upper
         THEN 'YES' ELSE 'no' END AS dataset_outlier
FROM mlops_demo_db.ml_training_runs r
CROSS JOIN bounds b
WHERE r.cost_usd < b.cost_lower OR r.cost_usd > b.cost_upper
   OR r.training_hours < b.hours_lower OR r.training_hours > b.hours_upper
   OR r.dataset_size < b.ds_lower OR r.dataset_size > b.ds_upper
ORDER BY r.cost_usd DESC;
