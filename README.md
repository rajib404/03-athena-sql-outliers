# Day 3: Athena & SQL for ML — Outlier Detection on S3 Data

Part of the **60-project MLOps series**. Builds on Day 2's Glue Data Catalog (`mlops_demo_db`) and same S3 bucket.

## What You'll Learn

- Create Athena external tables via DDL (vs Day 2's Crawler approach)
- Run SQL-based outlier detection on ML training data
- Six techniques: basic stats, Z-score, IQR, percentile, combined scoring, drift detection
- Query across projects through the shared Glue Data Catalog

## Architecture

```
S3 (ml_training_runs.csv)
  └── Glue Data Catalog (mlops_demo_db.ml_training_runs)
        └── Athena Workgroup (mlops-outlier-detection)
              └── 6 SQL Outlier Detection Queries
```

## Dataset

**500 rows** simulating ML experiment tracking with **25 seeded outliers** (5 per category):

| Category | What | Normal Range | Outlier Range |
|----------|------|-------------|---------------|
| Runaway cost | `cost_usd` | 5–500 | 2000–5000 |
| Perfect accuracy | `accuracy` | 0.55–0.95 | 0.999–1.0 |
| Long training | `training_hours` | 0.5–48 | 200–500 |
| Tiny dataset | `dataset_size` | 10K–500K | 5–50 |
| Impossible metrics | `accuracy` | positive | negative |

## Outlier Detection Queries

| # | Query | Technique | Educational Point |
|---|-------|-----------|-------------------|
| 1 | Basic Statistics | AVG, STDDEV, PERCENTILE | Understand distributions first |
| 2 | Z-Score | \|z\| > 3 | Classic method; CTE + CROSS JOIN |
| 3 | IQR | Tukey's fences | Robust for skewed data |
| 4 | Percentile | P01 / P99 | Distribution-agnostic |
| 5 | Combined Scoring | Multi-method + domain rules | Production pattern |
| 6 | Model Drift | LAG() window function | Sequential drift detection |

## Prerequisites

- AWS account with Athena access
- Day 2's Glue database (`mlops_demo_db`) — or this project will create it
- S3 bucket from Day 1/2
- Python 3.10+

## Setup

```bash
cd 03-athena-sql-outliers
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # Edit with your AWS credentials
```

## Run

```bash
python3 create_sample_data.py      # Generate 500-row CSV with 25 outliers
python3 upload_to_s3.py            # Upload to S3
python3 setup_athena.py            # Create workgroup + external table
python3 run_outlier_queries.py     # Run 6 outlier detection queries
python3 query_existing_tables.py   # Query Day 2's tables (optional)
python3 cleanup.py                 # Remove all Day 3 resources
```

## Key Concepts

**DDL vs Crawler**: Day 2 used a Glue Crawler to automatically infer schema. Day 3 defines the schema explicitly via `CREATE EXTERNAL TABLE`. Both register tables in the same Glue Data Catalog.

**Athena Workgroups**: Isolate query execution, control costs, and set output locations. Each project gets its own workgroup.

**Combined Scoring**: In production, no single outlier method catches everything. Query 5 combines Z-score, IQR, and domain-specific rules into an `outlier_score` with `likely_cause` classification.

## Project Structure

```
03-athena-sql-outliers/
├── .env.example              # AWS config template
├── .gitignore
├── requirements.txt
├── README.md
├── sql/                      # Reference SQL files
│   ├── 01_basic_stats.sql
│   ├── 02_zscore_outliers.sql
│   ├── 03_iqr_outliers.sql
│   ├── 04_percentile_outliers.sql
│   ├── 05_combined_flags.sql
│   └── 06_model_drift.sql
├── create_sample_data.py     # Step 1: Generate dataset
├── upload_to_s3.py           # Step 2: Upload to S3
├── setup_athena.py           # Step 3: Create workgroup + table
├── run_outlier_queries.py    # Step 4: Run outlier queries
├── query_existing_tables.py  # Step 5: Query Day 2 tables
└── cleanup.py                # Step 6: Tear down
```

## Cost

Athena charges $5/TB scanned. This dataset is ~40 KB, so all queries cost effectively $0.00.
