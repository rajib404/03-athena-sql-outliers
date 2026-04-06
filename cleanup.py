"""
Day 3 – Step 6: Clean up all resources created by this project.

Removes:
- Athena table ml_training_runs (via DROP TABLE)
- Athena workgroup mlops-outlier-detection
- S3 objects under athena-outliers/ and athena-results/ prefixes

Does NOT delete:
- Day 2's database (mlops_demo_db)
- Day 2's tables or S3 data (glue-crawler-demo/)
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

S3_PREFIXES_TO_DELETE = ["athena-outliers/", "athena-results/"]

athena_client = boto3.client("athena", region_name=REGION)
s3_client = boto3.client("s3", region_name=REGION)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def wait_for_query(query_execution_id: str, timeout: int = 60) -> dict:
    start = time.time()
    while True:
        resp = athena_client.get_query_execution(
            QueryExecutionId=query_execution_id
        )
        state = resp["QueryExecution"]["Status"]["State"]
        if state in ("SUCCEEDED", "FAILED", "CANCELLED"):
            return resp["QueryExecution"]
        if time.time() - start > timeout:
            break
        time.sleep(2)
    return resp["QueryExecution"]


# ---------------------------------------------------------------------------
# Cleanup steps
# ---------------------------------------------------------------------------

def drop_table() -> None:
    """Drop the ml_training_runs external table."""
    print("[1/3] Dropping Athena table ...")

    sql = f"DROP TABLE IF EXISTS {DATABASE_NAME}.ml_training_runs"

    try:
        resp = athena_client.start_query_execution(
            QueryString=sql,
            QueryExecutionContext={"Database": DATABASE_NAME},
            WorkGroup=WORKGROUP,
        )
        result = wait_for_query(resp["QueryExecutionId"])
        state = result["Status"]["State"]
        if state == "SUCCEEDED":
            print(f"  DROPPED  {DATABASE_NAME}.ml_training_runs")
        else:
            reason = result["Status"].get("StateChangeReason", "unknown")
            print(f"  WARN     DROP TABLE: {state} – {reason}")
    except ClientError as e:
        # Workgroup might already be deleted
        print(f"  WARN     {e.response['Error']['Message']}")
        print("           Trying via Glue API instead ...")
        try:
            glue_client = boto3.client("glue", region_name=REGION)
            glue_client.delete_table(
                DatabaseName=DATABASE_NAME, Name="ml_training_runs"
            )
            print(f"  DROPPED  {DATABASE_NAME}.ml_training_runs (via Glue)")
        except ClientError as e2:
            if "EntityNotFoundException" in str(e2):
                print(f"  SKIP     Table already removed")
            else:
                print(f"  WARN     {e2.response['Error']['Message']}")


def delete_workgroup() -> None:
    """Delete the Athena workgroup."""
    print("\n[2/3] Deleting Athena workgroup ...")

    try:
        athena_client.delete_work_group(
            WorkGroup=WORKGROUP,
            RecursiveDeleteOption=True,
        )
        print(f"  DELETED  Workgroup: {WORKGROUP}")
    except ClientError as e:
        if "not found" in str(e).lower():
            print(f"  SKIP     Workgroup already removed")
        else:
            print(f"  WARN     {e.response['Error']['Message']}")


def delete_s3_objects() -> None:
    """Remove S3 objects under athena-outliers/ and athena-results/."""
    print("\n[3/3] Removing S3 objects ...")

    total_deleted = 0
    for prefix in S3_PREFIXES_TO_DELETE:
        deleted = 0
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix=prefix):
                objects = page.get("Contents", [])
                if not objects:
                    continue
                delete_keys = [{"Key": obj["Key"]} for obj in objects]
                s3_client.delete_objects(
                    Bucket=BUCKET_NAME,
                    Delete={"Objects": delete_keys},
                )
                deleted += len(delete_keys)
        except ClientError as e:
            print(f"  WARN     {prefix}: {e.response['Error']['Message']}")
            continue

        if deleted > 0:
            print(f"  DELETED  s3://{BUCKET_NAME}/{prefix} ({deleted} objects)")
        else:
            print(f"  SKIP     s3://{BUCKET_NAME}/{prefix} (no objects)")
        total_deleted += deleted

    print(f"           Total deleted: {total_deleted} objects")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 6: Cleanup ===\n")
    print(f"  Bucket:    {BUCKET_NAME}")
    print(f"  Database:  {DATABASE_NAME} (NOT deleted – shared with Day 2)")
    print(f"  Workgroup: {WORKGROUP}")
    print()

    drop_table()
    delete_workgroup()
    delete_s3_objects()

    print(f"\n  DONE     Day 3 resources cleaned up")
    print(f"           Day 2 database & tables preserved\n")


if __name__ == "__main__":
    main()
