from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple


class BM25Retriever:
    """Класс для поиска объявлений по заголовкам и описаниям запросов с помощью BM25"""
    def __init__(self, k1: float = 2.0, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.title_bm25 = None
        self.desc_bm25 = None
        self.item_ids = None

    def fit(self, titles: List[str], descriptions: List[str], item_ids: List):
        """Строит BM25-индексы по заголовкам и описаниям и сохраняет идентификаторы объектов."""
        tokenized_titles = [t.split() for t in titles]
        tokenized_descs  = [d.split() for d in descriptions]

        self.title_bm25 = BM25Okapi(tokenized_titles, k1=self.k1, b=self.b)
        self.desc_bm25  = BM25Okapi(tokenized_descs,  k1=self.k1, b=self.b)
        self.item_ids = np.array(item_ids)

    def retrieve(
        self,
        query: str,
        top_k: int = 50,
        title_weight: float = 0.8,
        desc_weight: float = 0.2
    ) -> List[Tuple]:
        """Возвращает top-k наиболее релевантных объектов для заданного поискового запроса."""
        tokenized_query = query.split()

        title_scores = self.title_bm25.get_scores(tokenized_query)
        desc_scores  = self.desc_bm25.get_scores(tokenized_query)

        # Нормализуем скоры, чтобы они были сопоставимы
        title_norm = self._minmax_normalize(title_scores)
        desc_norm  = self._minmax_normalize(desc_scores)

        final_scores = title_weight * title_norm + desc_weight * desc_norm

        top_idx = np.argsort(final_scores)[::-1][:top_k]
        return list(zip(self.item_ids[top_idx], final_scores[top_idx]))

    def _minmax_normalize(self, scores: np.ndarray) -> np.ndarray:
        """Min-Max нормализация массива оценок релевантности на [0;1]."""
        min_s, max_s = scores.min(), scores.max()
        diff = max_s - min_s
        if diff == 0:
            return np.zeros_like(scores, dtype=np.float32)

        return (scores - min_s) / (max_s - min_s)
