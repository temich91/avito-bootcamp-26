from typing import Dict, Set, List
import numpy as np
import polars as pl
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.paths import *
from src.data.preprocessing import Preprocessor
from src.retrieval.bm25 import BM25Retriever


VAL_RATIO = 0.15
RANDOM_SEED = 42
TOP_K = 50
TITLE_WEIGHT = 0.8
DESC_WEIGHT = 0.2

# Названия колонок (поправь при необходимости)
TITLE_COL = "item_title_raw"
DESC_COL = "item_description_raw"
QUERY_COL = "search_query"
PARAMS_COL = "search_infm_params_text"
ITEM_ID_COL = "item_id"
QUERY_ID_COL = "query_id"


def create_query_id(df: pl.DataFrame) -> pl.DataFrame:
    """Создаёт стабильный query_id как хеш от текста запроса."""
    return df.with_columns(
        pl.col(QUERY_COL).hash().cast(pl.Utf8).alias(QUERY_ID_COL)
    )


def make_id_split(train_df: pl.DataFrame, val_ratio: float = 0.15, seed: int = 42):
    """Делит данные на train/val по уникальным query_id без утечки."""

    train_df = create_query_id(train_df)
    unique_queries = train_df[QUERY_ID_COL].unique().to_list()
    train_q, val_q = train_test_split(unique_queries, test_size=val_ratio, random_state=seed)

    train_part = train_df.filter(pl.col(QUERY_ID_COL).is_in(train_q))
    val_part = train_df.filter(pl.col(QUERY_ID_COL).is_in(val_q))

    print(f"Train queries: {len(train_q)} | Val queries: {len(val_q)}")
    print(f"Train pairs: {train_part.height} | Val pairs: {val_part.height}")
    return train_part, val_part


def build_ground_truth(df: pl.DataFrame) -> Dict[str, Set]:
    """Строит словарь {query_id: множество релевантных item_id}."""
    gt = (
        df.group_by(QUERY_ID_COL)
        .agg(pl.col(ITEM_ID_COL).unique().alias("items"))
        .to_dict(as_series=False)
    )
    return dict(zip(gt[QUERY_ID_COL], [set(x) for x in gt["items"]]))


def recall(predictions: Dict[str, List], ground_truth: Dict[str, Set], k: int = 50) -> float:
    """Считает средний Recall@k по всем запросам."""
    recalls = []
    for qid, preds in predictions.items():
        relevant = ground_truth.get(qid, set())
        if not relevant:
            continue
        hits = len(set(preds[:k]) & relevant)
        recalls.append(hits / len(relevant))
    return float(np.mean(recalls)) if recalls else 0.0


def main():
    print("Загрузка train...")
    train_raw = pl.read_parquet(TRAIN_PATH)
    train_raw = create_query_id(train_raw)

    train_df, val_df = make_id_split(train_raw, VAL_RATIO, RANDOM_SEED)
    val_query_ids = val_df[QUERY_ID_COL].unique().to_list()

    print("\nПрепроцессинг всего train...")
    preprocessor = Preprocessor(train_raw)

    text_cols = [QUERY_COL, PARAMS_COL, TITLE_COL, DESC_COL]
    processed = preprocessor.preprocess_text_features(
        text_cols,
        keep_cols=["item_id", "query_id"]
    )

    items_clean = (
        processed
        .unique(subset=[ITEM_ID_COL])
        .select([ITEM_ID_COL, f"clean_{TITLE_COL}", f"clean_{DESC_COL}"])
        .drop_nulls()
    )
    print(f"Уникальных объявлений для индекса: {items_clean.height}")

    print("\nОбучение BM25...")
    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(
        titles=items_clean[f"clean_{TITLE_COL}"].to_list(),
        descriptions=items_clean[f"clean_{DESC_COL}"].to_list(),
        item_ids=items_clean[ITEM_ID_COL].to_list(),
    )

    print("\nЛокальная валидация...")
    ground_truth = build_ground_truth(val_df)

    val_queries = (
        processed
        .filter(pl.col(QUERY_ID_COL).is_in(val_query_ids))
        .unique(subset=[QUERY_ID_COL])
    )

    val_predictions = {}
    for row in tqdm(val_queries.iter_rows(named=True), total=val_queries.height, desc="Val"):
        qid = row[QUERY_ID_COL]
        q_text = f"{row[f'clean_{QUERY_COL}']} {row.get(f'clean_{PARAMS_COL}', '')}".strip()
        ranked = retriever.retrieve(
            query=q_text,
            top_k=TOP_K,
            title_weight=TITLE_WEIGHT,
            desc_weight=DESC_WEIGHT,
        )
        val_predictions[qid] = [item_id for item_id, _ in ranked]

    val_recall = recall(val_predictions, ground_truth, k=TOP_K)
    print(f"\nLocal Recall@{TOP_K}: {val_recall:.4f}")


    print("\nЗагрузка бенчмарка...")
    queries = pl.read_parquet(QUERIES_PATH)
    items = pl.read_parquet(ITEMS_PATH)

    print("Препроцессинг benchmark_items + queries...")

    # Для items
    items_pre = Preprocessor(items)
    items_text_cols = [c for c in [TITLE_COL, DESC_COL] if c in items.columns]
    items_processed = items_pre.preprocess_text_features(items_text_cols, keep_cols=["item_id"])

    # Для queries
    queries_pre = Preprocessor(queries)
    queries_text_cols = [c for c in [QUERY_COL, PARAMS_COL] if c in queries.columns]
    queries_processed = queries_pre.preprocess_text_features(queries_text_cols, keep_cols=["query_id"])

    print("\nОбучение BM25 на benchmark_items...")
    items_for_index = (
        items_processed
        .unique(subset=[ITEM_ID_COL])
        .select([
            ITEM_ID_COL,
            f"clean_{TITLE_COL}",
            f"clean_{DESC_COL}",
        ])
        .drop_nulls()
    )

    retriever = BM25Retriever(k1=1.5, b=0.75)
    retriever.fit(
        titles=items_for_index[f"clean_{TITLE_COL}"].to_list(),
        descriptions=items_for_index[f"clean_{DESC_COL}"].to_list(),
        item_ids=items_for_index[ITEM_ID_COL].to_list(),
    )

    print("\nГенерация предсказаний...")
    results = []

    for row in tqdm(queries_processed.iter_rows(named=True), total=queries_processed.height, desc="Predict"):
        qid = row[QUERY_ID_COL]
        q_text = f"{row[f'clean_{QUERY_COL}']} {row.get(f'clean_{PARAMS_COL}', '')}".strip()

        ranked = retriever.retrieve(
            query=q_text,
            top_k=TOP_K,
            title_weight=TITLE_WEIGHT,
            desc_weight=DESC_WEIGHT,
        )
        results.append({
            "query_id": qid,
            "item_ids": [item_id for item_id, _ in ranked]
        })

    # Сохранение результатов в csv
    query_ids = []
    answers = []

    for row in results:
        qid = str(row["query_id"])
        item_ids = [str(iid) for iid in row["item_ids"][:50]]

        seen = set()
        unique_ids = []
        for iid in item_ids:
            if iid not in seen:
                seen.add(iid)
                unique_ids.append(iid)

        query_ids.append(qid)
        answers.append(" ".join(unique_ids))

    answer_df = pl.DataFrame({
        "query_id": query_ids,
        "answer": answers
    })

    answer_path = OUTPUT_DIR / "answer.csv"
    answer_df.write_csv(answer_path, separator=",")
    print(f"\nСабмит сохранён: {answer_path}")
    print(f"Строк в сабмите: {len(answer_df)}")


if __name__ == "__main__":
    main()
