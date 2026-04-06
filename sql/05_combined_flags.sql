-- 05_combined_flags.sql
-- Combined Multi-Method Outlier Scoring with Domain Rules
-- Production pattern: aggregate signals from multiple detection methods
-- Each flag adds to outlier_score; includes likely_cause classification

WITH stats AS (
    SELECT
        AVG(cost_usd) AS avg_cost, STDDEV(cost_usd) AS std_cost,
        AVG(accuracy) AS avg_acc, STDDEV(accuracy) AS std_acc,
        AVG(training_hours) AS avg_hours, STDDEV(training_hours) AS std_hours,
        AVG(CAST(dataset_size AS DOUBLE)) AS avg_ds, STDDEV(CAST(dataset_size AS DOUBLE)) AS std_ds,
        APPROX_PERCENTILE(cost_usd, 0.25) AS cost_q1,
        APPROX_PERCENTILE(cost_usd, 0.75) AS cost_q3,
        APPROX_PERCENTILE(training_hours, 0.25) AS hours_q1,
        APPROX_PERCENTILE(training_hours, 0.75) AS hours_q3
    FROM mlops_demo_db.ml_training_runs
),
scored AS (
    SELECT
        r.run_id,
        r.model_name,
        r.cost_usd,
        r.accuracy,
        r.loss,
        r.training_hours,
        r.dataset_size,
        -- Z-score flags (|z| > 3)
        CASE WHEN ABS(r.cost_usd - s.avg_cost) / NULLIF(s.std_cost, 0) > 3 THEN 1 ELSE 0 END
        + CASE WHEN ABS(r.accuracy - s.avg_acc) / NULLIF(s.std_acc, 0) > 3 THEN 1 ELSE 0 END
        + CASE WHEN ABS(r.training_hours - s.avg_hours) / NULLIF(s.std_hours, 0) > 3 THEN 1 ELSE 0 END
        -- IQR flags
        + CASE WHEN r.cost_usd > s.cost_q3 + 1.5 * (s.cost_q3 - s.cost_q1) THEN 1 ELSE 0 END
        + CASE WHEN r.training_hours > s.hours_q3 + 1.5 * (s.hours_q3 - s.hours_q1) THEN 1 ELSE 0 END
        -- Domain rules
        + CASE WHEN r.accuracy >= 0.999 THEN 1 ELSE 0 END
        + CASE WHEN r.accuracy < 0 THEN 1 ELSE 0 END
        + CASE WHEN r.dataset_size < 100 THEN 1 ELSE 0 END
        + CASE WHEN r.cost_usd > 1500 THEN 1 ELSE 0 END
        + CASE WHEN r.training_hours > 100 THEN 1 ELSE 0 END
        AS outlier_score,
        -- Likely cause classification
        CASE
            WHEN r.accuracy < 0 THEN 'logging_bug'
            WHEN r.accuracy >= 0.999 AND r.loss < 0.01 THEN 'data_leakage'
            WHEN r.cost_usd > 1500 THEN 'runaway_cost'
            WHEN r.training_hours > 100 THEN 'long_training'
            WHEN r.dataset_size < 100 THEN 'tiny_dataset'
            ELSE NULL
        END AS likely_cause
    FROM mlops_demo_db.ml_training_runs r
    CROSS JOIN stats s
)
SELECT *
FROM scored
WHERE outlier_score >= 2
ORDER BY outlier_score DESC, run_id;
