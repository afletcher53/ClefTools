import os
import pickle
from typing import List, Dict, Optional


import numpy as np
from pathlib import Path
import logging
from dotenv import load_dotenv
import csv

from Article import Article
from Vectorizer import Vectorizer

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(self, data_folder: Path):
        self.data_folder = data_folder

    def load_data(self, cd: str, folder: str) -> Dict[str, Article]:
        file_path = self.data_folder / folder / f"{cd}.csv"
        if not file_path.exists():
            raise FileNotFoundError(f"No data found for CD {
                                    cd} in the data folder.")

        articles = {}
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                article = Article(
                    pmid=row['PMID'],
                    title=row['title'],
                    abstract=row['abstract'],
                    label=int(row['label'])
                )
                articles[article.pmid] = article
        return articles


class DataController:
    def __init__(self, cd: str, source: str = 'clef_2017_train'):
        self.cd = cd
        self.source = source
        self.data_folder = Path('./data')
        self.data: Dict[str, Article] = {}
        self.seed = int(os.getenv('RANDOM_SEED', 42))
        self.data_loader = DataLoader(self.data_folder)
        self.vectorizer = None
        self._init_data()

    def _init_data(self):
        pickle_path = self.data_folder / self.source / f"{self.cd}_data.pkl"
        if pickle_path.exists():
            assert (self.data_folder / self.source / f"{self.cd}_vectorizer.pkl").exists(
            ), "Data and vectorizer parameter files must be present together."
            self._load_data()
        else:
            self.vectorizer = Vectorizer()
            self.data = self.data_loader.load_data(self.cd, self.source)
            self._make_data()
            self._save_data()

    def _make_data(self):
        np.random.seed(self.seed)
        pmids = list(self.data.keys())
        np.random.shuffle(pmids)
        self.data = {pmid: self.data[pmid] for pmid in pmids}
        self.data = self.vectorizer.vectorize(self.data)

    def _save_data(self):
        save_dir = self.data_folder / self.source
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            with open(save_dir / f"{self.cd}_data.pkl", 'wb') as f:
                pickle.dump(self.data, f)

            with open(save_dir / f"{self.cd}_vectorizer.pkl", 'wb') as f:
                pickle.dump(self.vectorizer, f)

            logger.info(f"Data and vectorizer saved for CD {self.cd}")
        except IOError as e:
            logger.error(f"Error saving data for CD {self.cd}: {e}")
            raise

    def _load_data(self):
        load_dir = self.data_folder / self.source

        try:
            with open(load_dir / f"{self.cd}_data.pkl", 'rb') as f:
                self.data = pickle.load(f)

            with open(load_dir / f"{self.cd}_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)

            logger.info(f"Data and vectorizer loaded for CD {self.cd}")
        except (IOError, pickle.UnpicklingError) as e:
            logger.error(f"Error loading data for CD {self.cd}: {e}")
            raise


if __name__ == "__main__":
    test = DataController("CD011134")
