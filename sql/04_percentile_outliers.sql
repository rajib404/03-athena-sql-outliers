-- 04_percentile_outliers.sql
-- Percentile-Based Outlier Detection: Distribution-agnostic
-- Technique: Flag values below 1st or above 99th percentile
-- Simplest method; works regardless of distribution shape

WITH pctiles AS (
    SELECT
        APPROX_PERCENTILE(cost_usd, 0.01) AS cost_p01,
        APPROX_PERCENTILE(cost_usd, 0.99) AS cost_p99,
        APPROX_PERCENTILE(accuracy, 0.01) AS acc_p01,
        APPROX_PERCENTILE(accuracy, 0.99) AS acc_p99,
        APPROX_PERCENTILE(training_hours, 0.01) AS hours_p01,
        APPROX_PERCENTILE(training_hours, 0.99) AS hours_p99,
        APPROX_PERCENTILE(dataset_size, 0.01) AS ds_p01,
        APPROX_PERCENTILE(dataset_size, 0.99) AS ds_p99
    FROM mlops_demo_db.ml_training_runs
)
SELECT
    r.run_id,
    r.model_name,
    r.cost_usd,
    r.accuracy,
    r.training_hours,
    r.dataset_size,
    CASE WHEN r.cost_usd < p.cost_p01 OR r.cost_usd > p.cost_p99 THEN 'YES' ELSE 'no' END AS cost_extreme,
    CASE WHEN r.accuracy < p.acc_p01 OR r.accuracy > p.acc_p99 THEN 'YES' ELSE 'no' END AS acc_extreme,
    CASE WHEN r.training_hours < p.hours_p01 OR r.training_hours > p.hours_p99 THEN 'YES' ELSE 'no' END AS hours_extreme,
    CASE WHEN r.dataset_size < p.ds_p01 OR r.dataset_size > p.ds_p99 THEN 'YES' ELSE 'no' END AS dataset_extreme
FROM mlops_demo_db.ml_training_runs r
CROSS JOIN pctiles p
WHERE r.cost_usd < p.cost_p01 OR r.cost_usd > p.cost_p99
   OR r.accuracy < p.acc_p01 OR r.accuracy > p.acc_p99
   OR r.training_hours < p.hours_p01 OR r.training_hours > p.hours_p99
   OR r.dataset_size < p.ds_p01 OR r.dataset_size > p.ds_p99
ORDER BY r.run_id;
