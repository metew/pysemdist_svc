from typing import Iterable, List
import numpy as np
from sentence_transformers import SentenceTransformer

_MODEL = None

def get_model(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    global _MODEL
    if _MODEL is None or (_MODEL and _MODEL._model_card not in model_name):
        _MODEL = SentenceTransformer(model_name)
    return _MODEL

def encode_texts(texts: Iterable[str], model_name: str = "all-MiniLM-L6-v2", normalize: bool = True) -> np.ndarray:
    model = get_model(model_name)
    emb = model.encode(list(texts), show_progress_bar=False, convert_to_numpy=True)
    if normalize:
        # L2 normalize for cosine-friendly HDBSCAN
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        emb = emb / norms
    return emb
