from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from .embeddings import encode_texts
from .config import settings

def _cluster_hdbscan(X: np.ndarray, min_cluster_size: int = 8, min_samples: Optional[int] = None, metric: str = "euclidean"):
    import hdbscan
    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples, metric=metric, cluster_selection_method="eom")
    labels = clusterer.fit_predict(X)
    # membership probabilities as confidence
    probs = getattr(clusterer, "probabilities_", None)
    if probs is None:
        probs = np.ones_like(labels, dtype=float)
    return labels, probs

def build_goal_sets(df: pd.DataFrame, *, text_col: str = "text", id_col: str = "id", model_name: str = "all-MiniLM-L6-v2", metric: str = "euclidean",
                    min_cluster_size: int = 8, min_samples: Optional[int] = None) -> List[Dict]:
    if df.empty:
        return []
    # Embed
    texts = df[text_col].fillna("").astype(str).tolist()
    X = encode_texts(texts, model_name=model_name, normalize=(metric == "cosine"))
    # Cluster
    labels, probs = _cluster_hdbscan(X, min_cluster_size=min_cluster_size, min_samples=min_samples, metric=("euclidean" if metric=="euclidean" else "euclidean"))
    # Note: HDBSCAN supports 'euclidean' well; for cosine, we rely on normalized vectors and euclidean distance approx.
    out: List[Dict] = []
    arr_ids = df[id_col].tolist()
    # group by label (ignore noise -1)
    from collections import defaultdict
    groups: Dict[int, List[int]] = defaultdict(list)
    for pid, lbl in zip(arr_ids, labels):
        if lbl == -1:  # noise
            continue
        groups[int(lbl)].append(int(pid))
    for lbl, ids in groups.items():
        out.append({"goal_id": f"hdbscan_{lbl}", "petition_ids": ids, "total_petitions": len(ids)})
    return out
