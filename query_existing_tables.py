"""
Day 3 – Step 5: Query Day 2's tables via Athena (bridge to previous project).

Demonstrates that the Glue Data Catalog is shared:
- SHOW TABLES to see all tables in mlops_demo_db
- Query Day 2's csv and json tables through Athena
- Shows Data Catalog integration across projects
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
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
DATABASE_NAME = os.getenv("GLUE_DATABASE_NAME", "mlops_demo_db")
WORKGROUP = os.getenv("ATHENA_WORKGROUP", "mlops-outlier-detection")

athena_client = boto3.client("athena", region_name=REGION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_query(query_execution_id: str, timeout: int = 120) -> dict:
    start = time.time()
    while True:
        resp = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return resp["QueryExecution"]
        if time.time() - start > timeout:
            print(f"  TIMEOUT  Query after {timeout}s")
            sys.exit(1)
        time.sleep(2)


def run_query(sql: str, description: str) -> tuple[dict, list[list[str]]]:
    """Execute query and return (execution_info, rows)."""
    try:
        resp = athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": DATABASE_NAME},
            WorkGroup=WORKGROUP,
        )
    except ClientError as e:
        print(f"  FAIL     {description}: {e.response['Error']['Message']}")
        return None, []

    result = wait_for_query(resp["QueryExecutionId"])
    state = result["Status"]["State"]

    if state != "SUCCEEDED":
        reason = result["Status"].get("StateChangeReason", "unknown")
        print(f"  FAIL     {description}: {state} – {reason}")
        return result, []

    rows = []
    paginator = athena_client.get_paginator("get_query_results")
    for page in paginator.paginate(QueryExecutionId=result["QueryExecutionId"]):
        for row in page["ResultSet"]["Rows"]:
            rows.append([col.get("VarCharValue", "") for col in row["Data"]])

    return result, rows


def print_table(rows: list[list[str]], max_rows: int = 10) -> None:
    if not rows or len(rows) < 2:
        print("    (no data)")
        return

    headers = rows[0]
    data = rows[1:]
    widths = [len(h) for h in headers]
    for row in data[:max_rows]:
        for i, val in enumerate(row):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(val)))

    header_line = "  | ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    separator = "-+-".join("-" * w for w in widths)
    print(f"    {header_line}")
    print(f"    {separator}")
    for row in data[:max_rows]:
        line = "  | ".join(
            str(row[i]).ljust(widths[i]) if i < len(row) else "".ljust(widths[i])
            for i in range(len(headers))
        )
        print(f"    {line}")
    if len(data) > max_rows:
        print(f"    ... ({len(data) - max_rows} more rows)")
    print(f"    [{len(data)} row(s)]")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 5: Query Day 2 Tables (Cross-Project) ===\n")

    # ---- 1. List all tables ------------------------------------------------
    print("[1/3] Listing all tables in database ...")
    _, rows = run_query(
        f"SHOW TABLES IN {DATABASE_NAME}",
        "SHOW TABLES"
    )
    if rows:
        tables = [r[0] for r in rows[1:]]  # skip header
        for t in tables:
            print(f"  TABLE    {DATABASE_NAME}.{t}")
        print()
    else:
        print("  WARN     No tables found or query failed")
        print("           Make sure Day 2 setup has been run.\n")
        return

    # ---- 2. Query Day 2 CSV table (sales_transactions / csv) ---------------
    csv_table = None
    for t in tables:
        if "csv" in t.lower() or "sales" in t.lower():
            csv_table = t
            break

    if csv_table:
        print(f"[2/3] Querying Day 2 CSV table: {csv_table} ...")
        _, rows = run_query(
            f"SELECT * FROM {DATABASE_NAME}.{csv_table} LIMIT 5",
            f"SELECT from {csv_table}"
        )
        print_table(rows)
        print()
    else:
        print("[2/3] No CSV table found from Day 2 (skipping)")
        print("      Expected tables from glue-crawler-demo/csv/\n")

    # ---- 3. Query Day 2 JSON table (model_metrics / json) ------------------
    json_table = None
    for t in tables:
        if "json" in t.lower() or "model" in t.lower() and t != "ml_training_runs":
            json_table = t
            break

    if json_table:
        print(f"[3/3] Querying Day 2 JSON table: {json_table} ...")
        _, rows = run_query(
            f"SELECT * FROM {DATABASE_NAME}.{json_table} LIMIT 5",
            f"SELECT from {json_table}"
        )
        print_table(rows)
        print()
    else:
        print("[3/3] No JSON table found from Day 2 (skipping)")
        print("      Expected tables from glue-crawler-demo/json/\n")

    print("  DONE     Cross-project queries complete\n")
    print("  NOTE     All tables share the same Glue Data Catalog database.")
    print("           Day 2 used a Crawler; Day 3 used DDL. Both approaches")
    print("           register tables in the same catalog.\n")


if __name__ == "__main__":
    main()
