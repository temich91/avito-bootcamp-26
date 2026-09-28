from pathlib import Path
from src.data.loader import load_train
import polars as pl
import re
from multiprocessing import Pool, cpu_count
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

def init_worker():
    global morph, STOP_WORDS
    morph = pymorphy3.MorphAnalyzer()
    STOP_WORDS = set(stopwords.words("russian"))

class Preprocessor:
    def __init__(self, train_df):
        self.raw_df = train_df

    def _normalize_text(self, text: str) -> str:
        """Очистка строки от спецсимволов и стоп-слов.
        Лемматизация не была применена чтобы ускорить обучение (нет GPU :( )
        """
        if not text:
            return ""
        text = re.sub(r'[^0-9а-яА-ЯёЁ\s]', ' ', str(text)).lower()
        tokens = [t for t in text.split() if t not in STOP_WORDS and len(t) > 1]
        return " ".join(tokens)

    def _process_series_parallel(self, series_data: list) -> list:
        """Обрабатывает список текстов в отдельном процессе"""
        return [self._normalize_text(text) for text in series_data]

    def preprocess_text_features(self, text_cols: list[str], keep_cols: list[str]) -> pl.DataFrame:
        """Параллельно нормализует текстовые признаки и возвращает датафрейм с новыми значениями."""
        if keep_cols is None:
            keep_cols = []

        cols_to_use = [c for c in text_cols + keep_cols if c in self.raw_df.columns]
        result_df = self.raw_df[cols_to_use]

        # кол-во доступных ядер процессора
        num_cores = cpu_count()

        for col in text_cols:
            print(f"Обработка {col}")
            raw_list = result_df[col].to_list()

            # обработка текстов батчами по числу ядер
            chunk_size = len(raw_list) // num_cores + 1
            chunks = [raw_list[i:i + chunk_size] for i in range(0, len(raw_list), chunk_size)]

            # параллельная обработка
            with Pool(processes=num_cores, initializer=init_worker) as pool:
                processed_chunks = pool.map(self._process_series_parallel, chunks)

            processed = [item for chunk in processed_chunks for item in chunk]

            result_df = result_df.with_columns(
                pl.Series(name=f"clean_{col}", values=processed)
            )

        result_df = result_df.drop(text_cols)

        return result_df

    def preprocess_numeric_features(self, numeric_cols):
        # В решении использовались только текстовые признаки
        pass
