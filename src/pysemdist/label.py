from __future__ import annotations
from typing import List, Tuple
import pandas as pd
from keybert import KeyBERT
from collections import Counter


def extract_keywords(texts: List[str], top_k: int = 10) -> List[str]:
    """Extract top keywords from a list of texts using KeyBERT."""
    if not texts:
        return []

    kw_model = KeyBERT()
    joined = "\n".join(texts)
    kws = kw_model.extract_keywords(
        joined,
        keyphrase_ngram_range=(1, 2),
        stop_words="english",
        top_n=top_k,
    )
    return [k for k, _ in kws]


def representative_texts(df_cluster: pd.DataFrame, n: int = 20) -> List[str]:
    """Return top representative texts in a cluster based on cluster_score."""
    samp = df_cluster.sort_values("cluster_score", ascending=False).head(n)
    return samp["text_norm"].tolist()


def label_cluster(df_cluster: pd.DataFrame) -> Tuple[str, List[str]]:
    """Generate a simple human-readable label and keywords for a cluster."""
    texts = representative_texts(df_cluster, n=min(30, len(df_cluster)))
    kws = extract_keywords(texts, top_k=10)

    # Simple label from most common keywords
    words = [w for kw in kws for w in kw.split()]
    common = [w for w, _ in Counter(words).most_common(3)]
    label = " ".join(common).title() if common else "General Issue"

    return label, kws
