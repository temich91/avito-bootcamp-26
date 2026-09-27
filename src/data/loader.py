import polars as pl
from typing import Tuple

def load_train(path: str) -> pl.DataFrame:
    return pl.read_parquet(path)

def load_benchmark(queries_path: str, items_path: str) -> Tuple[pl.DataFrame, pl.DataFrame]:
    queries = pl.read_parquet(queries_path)
    items = pl.read_parquet(items_path)
    return queries, items
