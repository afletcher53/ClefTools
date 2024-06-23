# article.py
from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Article:
    pmid: str
    title: str
    abstract: str
    label: int
    input_vector: Optional[np.ndarray] = None
