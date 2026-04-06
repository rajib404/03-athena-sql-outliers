"""
Day 3 – Step 3: Create Athena workgroup and external table.

- Creates Glue database if it doesn't exist (may have been cleaned up from Day 2)
- Creates workgroup 'mlops-outlier-detection' with results output location
- Runs CREATE EXTERNAL TABLE DDL (teaches DDL approach vs Day 2's Crawler)
- Verifies with SELECT COUNT(*)
- All operations are idempotent
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
OUTPUT_PREFIX = os.getenv("ATHENA_OUTPUT_PREFIX", "athena-results")
S3_DATA_PREFIX = "athena-outliers/training_runs"

OUTPUT_LOCATION = f"s3://{BUCKET_NAME}/{OUTPUT_PREFIX}/"

athena_client = boto3.client("athena", region_name=REGION)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_query(query_execution_id: str, timeout: int = 120) -> dict:
    """Poll until query completes or times out."""
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


def run_query(sql: str, description: str) -> dict:
    """Execute a query and wait for completion."""
    try:
        resp = athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": DATABASE_NAME},
            WorkGroup=WORKGROUP,
        )
    except ClientError as e:
        print(f"  FAIL     {description}: {e.response['Error']['Message']}")
        sys.exit(1)

    execution_id = resp["QueryExecutionId"]
    result = wait_for_query(execution_id)
    state = result["Status"]["State"]

    if state != "SUCCEEDED":
        reason = result["Status"].get("StateChangeReason", "unknown")
        print(f"  FAIL     {description}: {state} – {reason}")
        sys.exit(1)

    return result


def get_query_results(query_execution_id: str) -> list[list[str]]:
    """Fetch all result rows."""
    rows = []
    paginator = athena_client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=query_execution_id):
        for row in page["ResultSet"]["Rows"]:
            rows.append([col.get("VarCharValue", "") for col in row["Data"]])
    return rows


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def create_database() -> None:
    """Create Glue database if it doesn't exist (may have been cleaned up)."""
    print("[1/4] Ensuring Glue database exists ...")

    ddl = f"CREATE DATABASE IF NOT EXISTS {DATABASE_NAME}"

    # Need a temporary output location for this bootstrap query.
    # Workgroup might not exist yet, so use a direct OutputLocation.
    try:
        resp = athena_client.start_query_execution(
            QueryString=ddl,
            ResultConfiguration={"OutputLocation": OUTPUT_LOCATION},
        )
        result = wait_for_query(resp["QueryExecutionId"])
        state = result["Status"]["State"]
        if state == "SUCCEEDED":
            print(f"  CREATED  Database: {DATABASE_NAME}")
        else:
            reason = result["Status"].get("StateChangeReason", "unknown")
            print(f"  FAIL     CREATE DATABASE: {state} – {reason}")
            sys.exit(1)
    except ClientError as e:
        print(f"  FAIL     {e.response['Error']['Message']}")
        sys.exit(1)


def create_workgroup() -> None:
    """Create Athena workgroup with output location."""
    print("\n[2/4] Creating Athena workgroup ...")
    try:
        athena_client.create_work_group(
            Name=WORKGROUP,
            Configuration={
                "ResultConfiguration": {
                    "OutputLocation": OUTPUT_LOCATION,
                },
                "EnforceWorkGroupConfiguration": True,
                "PublishCloudWatchMetricsEnabled": False,
            },
            Description="MLOps Day 3 – Outlier detection with SQL",
        )
        print(f"  CREATED  Workgroup: {WORKGROUP}")
        print(f"           Output:    {OUTPUT_LOCATION}")
    except ClientError as e:
        if "already" in str(e).lower():
            print(f"  EXISTS   Workgroup: {WORKGROUP}")
        else:
            print(f"  FAIL     {e.response['Error']['Message']}")
            sys.exit(1)


def create_external_table() -> None:
    """Create external table via DDL (alternative to Glue Crawler)."""
    print("\n[3/4] Creating external table via DDL ...")

    ddl = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {DATABASE_NAME}.ml_training_runs (
        run_id           STRING,
        `timestamp`      STRING,
        model_name       STRING,
        framework        STRING,
        dataset_size     INT,
        training_hours   DOUBLE,
        gpu_memory_gb    INT,
        accuracy         DOUBLE,
        loss             DOUBLE,
        num_epochs       INT,
        learning_rate    DOUBLE,
        batch_size       INT,
        cost_usd         DOUBLE
    )
    ROW FORMAT DELIMITED
    FIELDS TERMINATED BY ','
    LINES TERMINATED BY '\\n'
    LOCATION 's3://{BUCKET_NAME}/{S3_DATA_PREFIX}/'
    TBLPROPERTIES ('skip.header.line.count'='1');
    """

    result = run_query(ddl, "CREATE TABLE")
    print(f"  CREATED  Table: {DATABASE_NAME}.ml_training_runs")
    print(f"           Data:  s3://{BUCKET_NAME}/{S3_DATA_PREFIX}/")


def verify_table() -> None:
    """Run SELECT COUNT(*) to verify data is accessible."""
    print("\n[4/4] Verifying table ...")

    sql = f"SELECT COUNT(*) AS row_count FROM {DATABASE_NAME}.ml_training_runs"
    result = run_query(sql, "SELECT COUNT(*)")

    execution_id = result["QueryExecutionId"]
    rows = get_query_results(execution_id)

    # rows[0] is header, rows[1] is data
    if len(rows) >= 2:
        count = rows[1][0]
        print(f"  VERIFY   Row count: {count}")
        stats = result.get("Statistics", {})
        scanned = stats.get("DataScannedInBytes", 0)
        print(f"           Data scanned: {scanned / 1024:.1f} KB")
    else:
        print("  WARN     Could not read row count")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 3: Setup Athena Workgroup & Table ===\n")

    create_database()
    create_workgroup()
    create_external_table()
    verify_table()

    print(f"\n  DONE     Athena ready for outlier queries\n")


if __name__ == "__main__":
    main()
