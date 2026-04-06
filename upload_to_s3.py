"""
Day 3 – Step 2: Upload training runs CSV to S3.

Uploads the generated CSV to s3://{bucket}/athena-outliers/training_runs/
(separate prefix from Day 2's glue-crawler-demo/).
"""

import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "my-ml-datalake-01")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")
S3_PREFIX = "athena-outliers/training_runs"
LOCAL_FILE = Path(__file__).parent / "sample_data" / "ml_training_runs.csv"

s3_client = boto3.client("s3", region_name=REGION)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 2: Upload Training Runs to S3 ===\n")

    if not LOCAL_FILE.exists():
        print(f"  MISSING  {LOCAL_FILE}")
        print("           Run create_sample_data.py first.")
        sys.exit(1)

    s3_key = f"{S3_PREFIX}/{LOCAL_FILE.name}"
    target = f"s3://{BUCKET_NAME}/{s3_key}"

    try:
        s3_client.upload_file(str(LOCAL_FILE), BUCKET_NAME, s3_key)
        print(f"  UPLOAD   {target}")
    except ClientError as e:
        print(f"  FAIL     {e.response['Error']['Message']}")
        sys.exit(1)

    # Verify
    try:
        resp = s3_client.head_object(Bucket=BUCKET_NAME, Key=s3_key)
        size_kb = resp["ContentLength"] / 1024
        print(f"  VERIFY   Size: {size_kb:.1f} KB")
    except ClientError:
        print("  WARN     Could not verify upload")

    print(f"\n  DONE     Data available at {target}\n")


if __name__ == "__main__":
    main()
