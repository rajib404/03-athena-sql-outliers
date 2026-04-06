"""
Day 3 – Step 1: Generate ML training runs dataset with seeded outliers.

Creates a 500-row CSV simulating ML experiment tracking data.
25 outliers (5 per category) are blended in to represent real MLOps failures:
  1. Runaway cost      – cost_usd 2000-5000
  2. Perfect accuracy  – accuracy 0.999-1.0, near-zero loss (data leakage)
  3. Long training     – training_hours 200-500
  4. Tiny dataset      – dataset_size 5-50
  5. Impossible metrics – negative accuracy (logging bugs)
"""

import csv
import random
from datetime import datetime, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SEED = 42
TOTAL_ROWS = 500
NUM_OUTLIERS_PER_CATEGORY = 5
OUTPUT_DIR = Path(__file__).parent / "sample_data"
OUTPUT_FILE = OUTPUT_DIR / "ml_training_runs.csv"

MODELS = [
    "resnet50", "bert-base", "gpt2-small", "efficientnet-b0",
    "yolov5s", "xgboost-v1", "random-forest-v2", "lstm-seq",
]
FRAMEWORKS = ["pytorch", "tensorflow", "sklearn", "xgboost"]

COLUMNS = [
    "run_id", "timestamp", "model_name", "framework", "dataset_size",
    "training_hours", "gpu_memory_gb", "accuracy", "loss", "num_epochs",
    "learning_rate", "batch_size", "cost_usd",
]


# ---------------------------------------------------------------------------
# Generators
# ---------------------------------------------------------------------------
def _random_timestamp(rng: random.Random) -> str:
    start = datetime(2024, 1, 1)
    offset = timedelta(days=rng.randint(0, 365), hours=rng.randint(0, 23),
                       minutes=rng.randint(0, 59))
    return (start + offset).strftime("%Y-%m-%d %H:%M:%S")


def _normal_row(rng: random.Random, run_id: int) -> dict:
    accuracy = round(rng.uniform(0.55, 0.95), 4)
    return {
        "run_id": f"run-{run_id:04d}",
        "timestamp": _random_timestamp(rng),
        "model_name": rng.choice(MODELS),
        "framework": rng.choice(FRAMEWORKS),
        "dataset_size": rng.randint(10_000, 500_000),
        "training_hours": round(rng.uniform(0.5, 48.0), 2),
        "gpu_memory_gb": rng.choice([8, 16, 24, 32, 40, 80]),
        "accuracy": accuracy,
        "loss": round(rng.uniform(0.05, 1.0 - accuracy + 0.05), 4),
        "num_epochs": rng.randint(5, 100),
        "learning_rate": rng.choice([0.1, 0.01, 0.001, 0.0001, 0.00001]),
        "batch_size": rng.choice([16, 32, 64, 128, 256]),
        "cost_usd": round(rng.uniform(5.0, 500.0), 2),
    }


def _outlier_runaway_cost(rng: random.Random, run_id: int) -> dict:
    row = _normal_row(rng, run_id)
    row["cost_usd"] = round(rng.uniform(2000.0, 5000.0), 2)
    return row


def _outlier_perfect_accuracy(rng: random.Random, run_id: int) -> dict:
    row = _normal_row(rng, run_id)
    row["accuracy"] = round(rng.uniform(0.999, 1.0), 4)
    row["loss"] = round(rng.uniform(0.0001, 0.005), 4)
    return row


def _outlier_long_training(rng: random.Random, run_id: int) -> dict:
    row = _normal_row(rng, run_id)
    row["training_hours"] = round(rng.uniform(200.0, 500.0), 2)
    return row


def _outlier_tiny_dataset(rng: random.Random, run_id: int) -> dict:
    row = _normal_row(rng, run_id)
    row["dataset_size"] = rng.randint(5, 50)
    return row


def _outlier_impossible_metrics(rng: random.Random, run_id: int) -> dict:
    row = _normal_row(rng, run_id)
    row["accuracy"] = round(rng.uniform(-0.5, -0.01), 4)
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("\n=== Day 3 · Step 1: Generate ML Training Runs Dataset ===\n")

    rng = random.Random(SEED)

    # Generate all rows
    rows: list[dict] = []
    outlier_ids: set[int] = set()

    # Pick 25 unique IDs for outliers, scattered across the dataset
    all_ids = list(range(1, TOTAL_ROWS + 1))
    rng.shuffle(all_ids)
    outlier_id_pool = all_ids[:NUM_OUTLIERS_PER_CATEGORY * 5]

    outlier_generators = [
        _outlier_runaway_cost,
        _outlier_perfect_accuracy,
        _outlier_long_training,
        _outlier_tiny_dataset,
        _outlier_impossible_metrics,
    ]

    # Assign 5 IDs to each outlier category
    outlier_map: dict[int, callable] = {}
    for i, gen in enumerate(outlier_generators):
        start = i * NUM_OUTLIERS_PER_CATEGORY
        end = start + NUM_OUTLIERS_PER_CATEGORY
        for oid in outlier_id_pool[start:end]:
            outlier_map[oid] = gen
            outlier_ids.add(oid)

    # Build rows
    for run_id in range(1, TOTAL_ROWS + 1):
        if run_id in outlier_map:
            rows.append(outlier_map[run_id](rng, run_id))
        else:
            rows.append(_normal_row(rng, run_id))

    # Sort by run_id (already ordered, but be explicit)
    rows.sort(key=lambda r: r["run_id"])

    # Write CSV
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    print(f"  CREATED  {OUTPUT_FILE}")
    print(f"           Total rows: {len(rows)}")
    print(f"           Outliers:   {len(outlier_ids)}")
    print()

    # Show outlier breakdown
    categories = [
        "runaway_cost", "perfect_accuracy", "long_training",
        "tiny_dataset", "impossible_metrics",
    ]
    for i, cat in enumerate(categories):
        start = i * NUM_OUTLIERS_PER_CATEGORY
        end = start + NUM_OUTLIERS_PER_CATEGORY
        ids = sorted(outlier_id_pool[start:end])
        id_strs = [f"run-{x:04d}" for x in ids]
        print(f"  OUTLIER  {cat:<22} -> {', '.join(id_strs)}")

    print(f"\n  DONE     {OUTPUT_FILE.name} ready for upload\n")


if __name__ == "__main__":
    main()
