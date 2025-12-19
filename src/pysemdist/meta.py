from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

def meta_cluster(centroids: pd.DataFrame, n_meta: int = 40) -> pd.DataFrame:
    """Cluster cluster-centroids into meta-themes. Always returns a DataFrame."""
    if centroids is None or centroids.empty:
        return pd.DataFrame(columns=["cluster_id", "meta_id"])
    X = np.vstack(centroids["centroid"].to_list())
    n = min(n_meta, len(centroids))
    if n <= 1:
        # nothing to cluster; put all into meta_id 0
        return pd.DataFrame({
            "cluster_id": centroids["cluster_id"].astype(int).to_list(),
            "meta_id": [0] * len(centroids),
        })
    model = AgglomerativeClustering(n_clusters=n, linkage="ward")
    labels = model.fit_predict(X)
    return pd.DataFrame({
        "cluster_id": centroids["cluster_id"].astype(int).to_list(),
        "meta_id": labels.astype(int).tolist(),
    })
