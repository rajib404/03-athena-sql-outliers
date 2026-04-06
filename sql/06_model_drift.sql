-- 06_model_drift.sql
-- Model Drift Detection using LAG() Window Function
-- Technique: Compare each run's metrics to previous run for same model
-- Cross-table analysis: queries Day 2's model_metrics JSON table

-- Part A: Drift detection on ml_training_runs (Day 3 data)
SELECT
    run_id,
    model_name,
    "timestamp",
    accuracy,
    loss,
    LAG(accuracy) OVER (PARTITION BY model_name ORDER BY "timestamp") AS prev_accuracy,
    LAG(loss) OVER (PARTITION BY model_name ORDER BY "timestamp") AS prev_loss,
    ROUND(accuracy - LAG(accuracy) OVER (PARTITION BY model_name ORDER BY "timestamp"), 4) AS accuracy_delta,
    ROUND(loss - LAG(loss) OVER (PARTITION BY model_name ORDER BY "timestamp"), 4) AS loss_delta,
    CASE
        WHEN ABS(accuracy - LAG(accuracy) OVER (PARTITION BY model_name ORDER BY "timestamp")) > 0.1
        THEN 'DRIFT'
        ELSE 'stable'
    END AS drift_flag
FROM mlops_demo_db.ml_training_runs
ORDER BY model_name, "timestamp";
