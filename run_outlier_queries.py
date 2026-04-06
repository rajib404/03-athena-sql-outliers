"""
Day 3 – Step 4: Run 6 outlier detection SQL queries against Athena.

Each query prints: name, description, SQL, formatted results, data scanned,
and execution time. Demonstrates progressively advanced SQL techniques for
identifying anomalous ML training runs.
"""

import os
import sys
import time

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "my-ml-datalake-01")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
DATABASE_NAME = os.getenv("GLUE_DATABASE_NAME", "mlops_demo_db")
WORKGROUP = os.getenv("ATHENA_WORKGROUP", "mlops-outlier-detection")

athena_client = boto3.client("athena", region_name=REGION)

# ---------------------------------------------------------------------------
# Athena helpers
# ---------------------------------------------------------------------------

def wait_for_query(query_execution_id: str, timeout: int = 120) -> dict:
    """Poll until query completes."""
    start = time.time()
    while True:
        resp = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return resp["QueryExecution"]
        if time.time() - start > timeout:
            print(f"  TIMEOUT  Query {query_execution_id} after {timeout}s")
            sys.exit(1)
        time.sleep(2)


def run_query(sql: str) -> tuple[dict, list[list[str]]]:
    """Execute query and return (execution_info, rows)."""
    resp = athena_client.start_query_execution(
        QueryString=sql,
        QueryExecutionContext={"Database": DATABASE_NAME},
        WorkGroup=WORKGROUP,
    )
    execution_id = resp["QueryExecutionId"]
    result = wait_for_query(execution_id)
    state = result["Status"]["State"]

    if state != "SUCCEEDED":
        reason = result["Status"].get("StateChangeReason", "unknown")
        return result, [[f"QUERY FAILED: {state} – {reason}"]]

    # Fetch results
    rows = []
    paginator = athena_client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=execution_id):
        for row in page["ResultSet"]["Rows"]:
            rows.append([col.get("VarCharValue", "") for col in row["Data"]])

    return result, rows


def print_results(rows: list[list[str]], max_rows: int = 30) -> None:
    """Pretty-print query results as a formatted table."""
    if not rows:
        print("    (no results)")
        return

    headers = rows[0]
    data = rows[1:]

    if not data:
        print("    (no data rows)")
        return

    # Calculate column widths
    widths = [len(h) for h in headers]
    display_data = data[:max_rows]
    for row in display_data:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(val)))

    # Print header
    header_line = "  | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)
    print(f"    {header_line}")
    print(f"    {separator}")

    # Print data
    for row in display_data:
        line = "  | ".join(
            str(row[i]).ljust(widths[i]) if i < len(row) else "".ljust(widths[i])
            for i in range(len(headers))
        )
        print(f"    {line}")

    if len(data) > max_rows:
        print(f"    ... ({len(data) - max_rows} more rows)")

    print(f"    [{len(data)} row(s)]")


def print_stats(result: dict) -> None:
    """Print query execution statistics."""
    stats = result.get("Statistics", {})
    scanned_bytes = stats.get("DataScannedInBytes", 0)
    exec_ms = stats.get("EngineExecutionTimeInMillis", 0)

    if scanned_bytes < 1024:
        scanned_str = f"{scanned_bytes} B"
    elif scanned_bytes < 1024 * 1024:
        scanned_str = f"{scanned_bytes / 1024:.1f} KB"
    else:
        scanned_str = f"{scanned_bytes / (1024*1024):.2f} MB"

    print(f"    Data scanned: {scanned_str}  |  Execution time: {exec_ms} ms")


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

QUERIES = [
    {
        "name": "1. Basic Statistics",
        "description": (
            "Understand distributions before detecting outliers.\n"
            "    Technique: AVG, STDDEV, MIN, MAX, APPROX_PERCENTILE"
        ),
        "sql": f"""
SELECT
    'cost_usd' AS metric,
    COUNT(cost_usd) AS cnt,
    ROUND(AVG(cost_usd), 2) AS mean,
    ROUND(STDDEV(cost_usd), 2) AS stddev,
    MIN(cost_usd) AS min_val,
    MAX(cost_usd) AS max_val,
    ROUND(APPROX_PERCENTILE(cost_usd, 0.50), 2) AS median
FROM {DATABASE_NAME}.ml_training_runs

UNION ALL

SELECT
    'accuracy', COUNT(accuracy),
    ROUND(AVG(accuracy), 4), ROUND(STDDEV(accuracy), 4),
    MIN(accuracy), MAX(accuracy),
    ROUND(APPROX_PERCENTILE(accuracy, 0.50), 4)
FROM {DATABASE_NAME}.ml_training_runs

UNION ALL

SELECT
    'training_hours', COUNT(training_hours),
    ROUND(AVG(training_hours), 2), ROUND(STDDEV(training_hours), 2),
    MIN(training_hours), MAX(training_hours),
    ROUND(APPROX_PERCENTILE(training_hours, 0.50), 2)
FROM {DATABASE_NAME}.ml_training_runs

UNION ALL

SELECT
    'dataset_size', COUNT(dataset_size),
    ROUND(AVG(dataset_size), 2), ROUND(STDDEV(dataset_size), 2),
    MIN(dataset_size), MAX(dataset_size),
    ROUND(APPROX_PERCENTILE(dataset_size, 0.50), 2)
FROM {DATABASE_NAME}.ml_training_runs
""",
    },
    {
        "name": "2. Z-Score Outliers",
        "description": (
            "Classic statistical method: flag values > 3 standard deviations from mean.\n"
            "    Technique: CTE for stats + CROSS JOIN pattern"
        ),
        "sql": f"""
WITH stats AS (
    SELECT
        AVG(cost_usd) AS avg_cost, STDDEV(cost_usd) AS std_cost,
        AVG(accuracy) AS avg_acc, STDDEV(accuracy) AS std_acc,
        AVG(training_hours) AS avg_hours, STDDEV(training_hours) AS std_hours,
        AVG(CAST(dataset_size AS DOUBLE)) AS avg_ds, STDDEV(CAST(dataset_size AS DOUBLE)) AS std_ds
    FROM {DATABASE_NAME}.ml_training_runs
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
FROM {DATABASE_NAME}.ml_training_runs r
CROSS JOIN stats s
WHERE ABS(r.cost_usd - s.avg_cost) / NULLIF(s.std_cost, 0) > 3
   OR ABS(r.accuracy - s.avg_acc) / NULLIF(s.std_acc, 0) > 3
   OR ABS(r.training_hours - s.avg_hours) / NULLIF(s.std_hours, 0) > 3
   OR ABS(CAST(r.dataset_size AS DOUBLE) - s.avg_ds) / NULLIF(s.std_ds, 0) > 3
ORDER BY cost_zscore DESC
""",
    },
    {
        "name": "3. IQR Outliers",
        "description": (
            "Tukey's fences: more robust than Z-score for skewed distributions.\n"
            "    Technique: value outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]"
        ),
        "sql": f"""
WITH quartiles AS (
    SELECT
        APPROX_PERCENTILE(cost_usd, 0.25) AS cost_q1,
        APPROX_PERCENTILE(cost_usd, 0.75) AS cost_q3,
        APPROX_PERCENTILE(training_hours, 0.25) AS hours_q1,
        APPROX_PERCENTILE(training_hours, 0.75) AS hours_q3,
        APPROX_PERCENTILE(CAST(dataset_size AS DOUBLE), 0.25) AS ds_q1,
        APPROX_PERCENTILE(CAST(dataset_size AS DOUBLE), 0.75) AS ds_q3
    FROM {DATABASE_NAME}.ml_training_runs
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
    CASE WHEN r.cost_usd < b.cost_lower OR r.cost_usd > b.cost_upper THEN 'YES' ELSE 'no' END AS cost_outlier,
    CASE WHEN r.training_hours < b.hours_lower OR r.training_hours > b.hours_upper THEN 'YES' ELSE 'no' END AS hours_outlier,
    CASE WHEN CAST(r.dataset_size AS DOUBLE) < b.ds_lower OR CAST(r.dataset_size AS DOUBLE) > b.ds_upper THEN 'YES' ELSE 'no' END AS dataset_outlier
FROM {DATABASE_NAME}.ml_training_runs r
CROSS JOIN bounds b
WHERE r.cost_usd < b.cost_lower OR r.cost_usd > b.cost_upper
   OR r.training_hours < b.hours_lower OR r.training_hours > b.hours_upper
   OR CAST(r.dataset_size AS DOUBLE) < b.ds_lower OR CAST(r.dataset_size AS DOUBLE) > b.ds_upper
ORDER BY r.cost_usd DESC
""",
    },
    {
        "name": "4. Percentile Outliers",
        "description": (
            "Distribution-agnostic: flag values below 1st or above 99th percentile.\n"
            "    Technique: Simplest outlier detection method"
        ),
        "sql": f"""
WITH pctiles AS (
    SELECT
        APPROX_PERCENTILE(cost_usd, 0.01) AS cost_p01,
        APPROX_PERCENTILE(cost_usd, 0.99) AS cost_p99,
        APPROX_PERCENTILE(accuracy, 0.01) AS acc_p01,
        APPROX_PERCENTILE(accuracy, 0.99) AS acc_p99,
        APPROX_PERCENTILE(training_hours, 0.01) AS hours_p01,
        APPROX_PERCENTILE(training_hours, 0.99) AS hours_p99,
        APPROX_PERCENTILE(CAST(dataset_size AS DOUBLE), 0.01) AS ds_p01,
        APPROX_PERCENTILE(CAST(dataset_size AS DOUBLE), 0.99) AS ds_p99
    FROM {DATABASE_NAME}.ml_training_runs
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
    CASE WHEN CAST(r.dataset_size AS DOUBLE) < p.ds_p01 OR CAST(r.dataset_size AS DOUBLE) > p.ds_p99 THEN 'YES' ELSE 'no' END AS dataset_extreme
FROM {DATABASE_NAME}.ml_training_runs r
CROSS JOIN pctiles p
WHERE r.cost_usd < p.cost_p01 OR r.cost_usd > p.cost_p99
   OR r.accuracy < p.acc_p01 OR r.accuracy > p.acc_p99
   OR r.training_hours < p.hours_p01 OR r.training_hours > p.hours_p99
   OR CAST(r.dataset_size AS DOUBLE) < p.ds_p01 OR CAST(r.dataset_size AS DOUBLE) > p.ds_p99
ORDER BY r.run_id
""",
    },
    {
        "name": "5. Combined Scoring",
        "description": (
            "Production pattern: aggregate multiple detection methods into outlier_score.\n"
            "    Technique: Z-score + IQR + domain rules -> likely_cause classification"
        ),
        "sql": f"""
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
    FROM {DATABASE_NAME}.ml_training_runs
),
scored AS (
    SELECT
        r.run_id, r.model_name, r.cost_usd, r.accuracy, r.loss,
        r.training_hours, r.dataset_size,
        CASE WHEN ABS(r.cost_usd - s.avg_cost) / NULLIF(s.std_cost, 0) > 3 THEN 1 ELSE 0 END
        + CASE WHEN ABS(r.accuracy - s.avg_acc) / NULLIF(s.std_acc, 0) > 3 THEN 1 ELSE 0 END
        + CASE WHEN ABS(r.training_hours - s.avg_hours) / NULLIF(s.std_hours, 0) > 3 THEN 1 ELSE 0 END
        + CASE WHEN r.cost_usd > s.cost_q3 + 1.5 * (s.cost_q3 - s.cost_q1) THEN 1 ELSE 0 END
        + CASE WHEN r.training_hours > s.hours_q3 + 1.5 * (s.hours_q3 - s.hours_q1) THEN 1 ELSE 0 END
        + CASE WHEN r.accuracy >= 0.999 THEN 1 ELSE 0 END
        + CASE WHEN r.accuracy < 0 THEN 1 ELSE 0 END
        + CASE WHEN r.dataset_size < 100 THEN 1 ELSE 0 END
        + CASE WHEN r.cost_usd > 1500 THEN 1 ELSE 0 END
        + CASE WHEN r.training_hours > 100 THEN 1 ELSE 0 END
        AS outlier_score,
        CASE
            WHEN r.accuracy < 0 THEN 'logging_bug'
            WHEN r.accuracy >= 0.999 AND r.loss < 0.01 THEN 'data_leakage'
            WHEN r.cost_usd > 1500 THEN 'runaway_cost'
            WHEN r.training_hours > 100 THEN 'long_training'
            WHEN r.dataset_size < 100 THEN 'tiny_dataset'
            ELSE NULL
        END AS likely_cause
    FROM {DATABASE_NAME}.ml_training_runs r
    CROSS JOIN stats s
)
SELECT *
FROM scored
WHERE outlier_score >= 2
ORDER BY outlier_score DESC, run_id
""",
    },
    {
        "name": "6. Model Drift Detection",
        "description": (
            "Compare each run's metrics to previous run for the same model.\n"
            "    Technique: LAG() window function for sequential drift detection"
        ),
        "sql": f"""
WITH sequenced AS (
    SELECT
        run_id,
        model_name,
        timestamp,
        accuracy,
        loss,
        LAG(accuracy) OVER (PARTITION BY model_name ORDER BY timestamp) AS prev_accuracy,
        LAG(loss) OVER (PARTITION BY model_name ORDER BY timestamp) AS prev_loss
    FROM {DATABASE_NAME}.ml_training_runs
)
SELECT
    run_id,
    model_name,
    timestamp,
    ROUND(accuracy, 4) AS accuracy,
    ROUND(prev_accuracy, 4) AS prev_accuracy,
    ROUND(accuracy - prev_accuracy, 4) AS accuracy_delta,
    ROUND(loss, 4) AS loss,
    ROUND(prev_loss, 4) AS prev_loss,
    ROUND(loss - prev_loss, 4) AS loss_delta,
    CASE
        WHEN prev_accuracy IS NULL THEN 'first_run'
        WHEN ABS(accuracy - prev_accuracy) > 0.1 THEN 'DRIFT'
        ELSE 'stable'
    END AS drift_flag
FROM sequenced
WHERE prev_accuracy IS NOT NULL
  AND ABS(accuracy - prev_accuracy) > 0.1
ORDER BY ABS(accuracy - prev_accuracy) DESC
""",
    },
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 4: Outlier Detection SQL Queries ===\n")

    for i, query in enumerate(QUERIES):
        print(f"{'='*70}")
        print(f"  Query {query['name']}")
        print(f"  {query['description']}")
        print(f"{'='*70}")

        # Show SQL (trimmed)
        sql_display = query["sql"].strip()
        if len(sql_display) > 500:
            sql_display = sql_display[:500] + "\n    ..."
        print(f"\n  SQL:\n    {sql_display.replace(chr(10), chr(10) + '    ')}\n")

        # Execute
        print("  Running ...", end="", flush=True)
        try:
            result, rows = run_query(query["sql"])
        except ClientError as e:
            print(f"\n  FAIL     {e.response['Error']['Message']}")
            continue

        state = result["Status"]["State"]
        if state != "SUCCEEDED":
            print(f"\n  FAIL     {state}")
            continue

        print(f" done.\n")

        # Results
        print("  Results:")
        print_results(rows)
        print()

        # Stats
        print_stats(result)
        print()

    print(f"\n  DONE     All {len(QUERIES)} queries executed\n")


if __name__ == "__main__":
    main()
