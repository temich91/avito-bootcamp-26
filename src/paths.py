from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = SRC_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.parquet"
QUERIES_PATH = DATA_DIR / "benchmark_queries.parquet"
ITEMS_PATH = DATA_DIR / "benchmark_items.parquet"
