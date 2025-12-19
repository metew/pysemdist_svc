from __future__ import annotations
from typing import Iterable
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


def embed_texts(
    texts: Iterable[str],
    model_name: str = "intfloat/e5-base-v2",
    batch_size: int = 128,
) -> np.ndarray:
    """Embed a list of texts using a SentenceTransformer model."""
    model = SentenceTransformer(model_name)

    # For e5 models: prepend "query: " or "passage: " as needed.
    # Petitions are treated as passages.
    def prep(t: str) -> str:
        return f"passage: {t}" if "e5" in model_name else t

    X = []
    buf = []

    for t in texts:
        buf.append(prep(t))
        if len(buf) >= batch_size:
            X.append(model.encode(buf, normalize_embeddings=True))
            buf = []

    # Encode remaining texts
    if buf:
        X.append(model.encode(buf, normalize_embeddings=True))

    return np.vstack(X) if X else np.zeros((0, 768), dtype=np.float32)
