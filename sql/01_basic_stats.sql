-- 01_basic_stats.sql
-- Basic Statistics: Understand distributions before detecting outliers
-- Technique: AVG, STDDEV, MIN, MAX, APPROX_PERCENTILE

SELECT
    'cost_usd' AS metric,
    COUNT(cost_usd) AS cnt,
    ROUND(AVG(cost_usd), 2) AS mean,
    ROUND(STDDEV(cost_usd), 2) AS stddev,
    MIN(cost_usd) AS min_val,
    MAX(cost_usd) AS max_val,
    ROUND(APPROX_PERCENTILE(cost_usd, 0.25), 2) AS p25,
    ROUND(APPROX_PERCENTILE(cost_usd, 0.50), 2) AS p50,
    ROUND(APPROX_PERCENTILE(cost_usd, 0.75), 2) AS p75
FROM mlops_demo_db.ml_training_runs

UNION ALL

SELECT
    'accuracy' AS metric,
    COUNT(accuracy) AS cnt,
    ROUND(AVG(accuracy), 4) AS mean,
    ROUND(STDDEV(accuracy), 4) AS stddev,
    MIN(accuracy) AS min_val,
    MAX(accuracy) AS max_val,
    ROUND(APPROX_PERCENTILE(accuracy, 0.25), 4) AS p25,
    ROUND(APPROX_PERCENTILE(accuracy, 0.50), 4) AS p50,
    ROUND(APPROX_PERCENTILE(accuracy, 0.75), 4) AS p75
FROM mlops_demo_db.ml_training_runs

UNION ALL

SELECT
    'training_hours' AS metric,
    COUNT(training_hours) AS cnt,
    ROUND(AVG(training_hours), 2) AS mean,
    ROUND(STDDEV(training_hours), 2) AS stddev,
    MIN(training_hours) AS min_val,
    MAX(training_hours) AS max_val,
    ROUND(APPROX_PERCENTILE(training_hours, 0.25), 2) AS p25,
    ROUND(APPROX_PERCENTILE(training_hours, 0.50), 2) AS p50,
    ROUND(APPROX_PERCENTILE(training_hours, 0.75), 2) AS p75
FROM mlops_demo_db.ml_training_runs

UNION ALL

SELECT
    'dataset_size' AS metric,
    COUNT(dataset_size) AS cnt,
    ROUND(AVG(dataset_size), 2) AS mean,
    ROUND(STDDEV(dataset_size), 2) AS stddev,
    MIN(dataset_size) AS min_val,
    MAX(dataset_size) AS max_val,
    ROUND(APPROX_PERCENTILE(dataset_size, 0.25), 2) AS p25,
    ROUND(APPROX_PERCENTILE(dataset_size, 0.50), 2) AS p50,
    ROUND(APPROX_PERCENTILE(dataset_size, 0.75), 2) AS p75
FROM mlops_demo_db.ml_training_runs;
