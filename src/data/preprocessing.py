from pathlib import Path
from loader import load_train
import polars as pl
import re
from multiprocessing import Pool, cpu_count
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import pymorphy3

def init_worker():
    global morph, STOP_WORDS
    morph = pymorphy3.MorphAnalyzer()
    STOP_WORDS = set(stopwords.words("russian"))

class Preprocessor:
    def __init__(self, raw_path):
        self.raw_df = load_train(raw_path)[:200]

    def _normalize_text(self, text: str) -> str:
        """
        Нормализация заданного текста.
        """
        if text is None:
            return ""

        # Очистка от спец символов
        cleaned =  re.sub(r'[^0-9а-яА-ЯёЁ\s]', '', text)

        # Перевод в нижний регистр
        lowered = cleaned.lower()

        # Токенизация
        tokens = word_tokenize(lowered, language="russian")

        # Удаление стоп-слов
        morph = pymorphy3.MorphAnalyzer()
        lemmas = []
        for tok in tokens:
            if tok not in STOP_WORDS:
                lemma = morph.parse(tok)[0].normal_form
                lemmas.append(lemma)

        return " ".join(lemmas)

    def _process_series_parallel(self, series_data: list) -> list:
        """Обрабатывает список текстов в отдельном процессе"""
        return [self._normalize_text(text) for text in series_data]

    def preprocess_text_features(self, text_cols: list[str]) -> pl.DataFrame:
        """
        Параллельно нормализует текстовые признаки и возвращает датафрейм с новыми значениями.
        """
        result_df = self.raw_df[text_cols]

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

        return result_df.drop(text_cols)

    def preprocess_numeric_features(self, numeric_cols):
        pass

# Вид услуги убрать
if __name__ == "__main__":
    p = Preprocessor(Path("../../data/train.parquet"))
    print(p.preprocess_text_features(["search_query", "item_description_raw"]))
