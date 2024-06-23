
from pathlib import Path
from typing import Dict
import gensim
import gensim.downloader as api
from gensim.utils import simple_preprocess
import logging
from dotenv import load_dotenv
import numpy as np

from Article import Article

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Vectorizer:
    def __init__(self, model_name: str = 'fasttext-wiki-news-subwords-300'):
        self.model_name = model_name
        self.model_path = Path(f'./models/{model_name}.bin')
        self.model = None

    def _load_model(self):
        if self.model is None:
            if not self.model_path.exists():
                logger.info(f"FastText model not found. Downloading {
                            self.model_name}...")
                self.model_path.parent.mkdir(parents=True, exist_ok=True)
                model = api.load(self.model_name)
                model.save_word2vec_format(str(self.model_path), binary=True)
                logger.info("Download completed and model saved.")
            else:
                logger.info("Loading existing FastText model...")
            self.model = gensim.models.KeyedVectors.load_word2vec_format(
                str(self.model_path), binary=True)

    def vectorize(self, articles: Dict[str, Article]) -> Dict[str, Article]:
        self._load_model()
        for pmid, article in articles.items():
            title_vector = self._text_to_vector(article.title)
            abstract_vector = self._text_to_vector(article.abstract)
            article.input_vector = np.concatenate(
                [title_vector, abstract_vector])
        return articles

    def _text_to_vector(self, text: str) -> np.ndarray:
        words = simple_preprocess(text)
        word_vectors = [self.model[word]
                        for word in words if word in self.model]
        return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(self.model.vector_size)

    def get_params(self):
        return {
            'model_name': self.model_name,
            'model_path': str(self.model_path),
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key == 'model_path':
                self.model_path = Path(value)
            else:
                setattr(self, key, value)
        return self

    def __getstate__(self):
        # Custom pickling method
        state = self.__dict__.copy()
        # Don't pickle 'model' as it's large and can be reloaded
        state['model'] = None
        return state

    def __setstate__(self, state):
        # Custom unpickling method
        self.__dict__.update(state)
        # Reload the model when unpickling
        self._load_model()
