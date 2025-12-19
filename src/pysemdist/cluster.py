from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


# ===============================
# HDBSCAN-based clustering (core)
# ===============================

def _fallback_kmeans(df: pd.DataFrame, X: np.ndarray, *, target_k: int = 8) -> pd.DataFrame:
    from sklearn.decomposition import PCA
    from sklearn.cluster import MiniBatchKMeans
    # light PCA for speed/stability
    n, d = X.shape
    n_comp = min(64, max(2, min(n, d) - 1))
    Xp = X if n_comp < 2 else PCA(n_components=n_comp, svd_solver="randomized", random_state=42).fit_transform(X)
    k = max(2, min(target_k, n))
    km = MiniBatchKMeans(n_clusters=k, random_state=42, batch_size=2048)
    labels = km.fit_predict(Xp)
    centers = km.cluster_centers_
    # distance to centroid (negative → higher is better when we sort desc)
    from numpy.linalg import norm
    dists = np.array([norm(x - centers[l]) for x, l in zip(Xp, labels)])
    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    out["cluster_score"] = -dists.astype(float)
    return out

def cluster_embeddings_hdbscan(
    df: pd.DataFrame,
    *,
    embedding_col: str = "embedding",
    min_cluster_size: int = 8,
    min_samples: int | None = None,
    metric: str = "euclidean",                 # "euclidean" or "cosine"
    cluster_selection_method: str = "eom",     # "eom" (broad) or "leaf" (fine-grained)
    cluster_selection_epsilon: float = 0.0,    # optional epsilon cut
    allow_singleton_noise: bool = False,        # map -1 (noise) to singleton cluster ids
    pca_n_components: int | None = None,       # optional PCA prior to clustering
    pca_random_state: int = 42,
) -> pd.DataFrame:
    """
    HDBSCAN clustering over petition embeddings with robust metric handling.

    - If metric == "cosine" and n <= ~5000, uses a precomputed cosine distance matrix.
    - Otherwise, uses Euclidean on L2-normalized vectors (≈ cosine ranking) to keep it scalable.
    - cluster_selection_method: "eom" (stable coarse themes) or "leaf" (finer, more clusters).
    - Optionally applies PCA before clustering (re-normalizes if cosine-like behavior is desired).

    Returns a copy of df with:
      - cluster_id (ints; noise either kept as -1 or converted to unique ids)
      - cluster_score (HDBSCAN membership probability in [0, 1])
    """
    import hdbscan
    import numpy as np
    from sklearn.decomposition import PCA
    from sklearn.metrics.pairwise import cosine_distances

    if df.empty:
        return df.assign(cluster_id=[], cluster_score=[])

    # Stack embeddings
    X = np.vstack(df[embedding_col].to_list()).astype(np.float32)
    n = X.shape[0]

    # Optional PCA for speed/denoise
    if pca_n_components:
        n_samples, n_features = X.shape
        n_comp = min(int(pca_n_components), max(2, min(n_samples, n_features) - 1))
        if n_comp >= 2:
            pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=pca_random_state)
            X = pca.fit_transform(X)

    # Normalize to unit length so Euclidean ≈ Cosine (important when we fallback)
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    X = X / norms

    # HDBSCAN params
    mcs = max(2, min(int(min_cluster_size), n))
    ms = mcs if (min_samples is None) else max(1, min(int(min_samples), n))

    # Strategy: exact cosine via precomputed distances ONLY for small n
    PRECOMPUTED_MAX_N = 5000
    use_precomputed = (metric.lower() == "cosine" and n <= PRECOMPUTED_MAX_N)

    if use_precomputed:
        D = cosine_distances(X)  # O(n^2) memory — OK only for small n
        model = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric="precomputed",
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=float(cluster_selection_epsilon),
            prediction_data=False,
        )
        labels = model.fit_predict(D).astype(int)
    else:
        # Fall back to Euclidean on L2-normalized vectors (cosine-like)
        actual_metric = "euclidean"
        model = hdbscan.HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            metric=actual_metric,
            cluster_selection_method=cluster_selection_method,
            cluster_selection_epsilon=float(cluster_selection_epsilon),
            prediction_data=False,
        )
        labels = model.fit_predict(X).astype(int)

    probs = getattr(model, "probabilities_", None)
    if probs is None:
        probs = np.ones(n, dtype=float)

    out = df.copy()
    out["cluster_id"] = labels
    out["cluster_score"] = probs.astype(float)

    # Noise handling: either keep -1 or convert to singleton clusters
    if (labels == -1).any():
        if allow_singleton_noise:
            # Assign unique ids after the max non-noise id
            non_noise = out["cluster_id"] >= 0
            start = int(out.loc[non_noise, "cluster_id"].max()) + 1 if non_noise.any() else 0
            noise_idx = np.where(labels == -1)[0]
            for j, i in enumerate(noise_idx):
                out.iloc[i, out.columns.get_loc("cluster_id")] = start + j
        # else: keep -1 as explicit noise

    # Reindex non-negative ids to 0..K-1 for cleanliness (leave -1 untouched if kept)
    if allow_singleton_noise:
        # all ids are non-negative now
        uniq = sorted(out["cluster_id"].unique().tolist())
        remap = {old: i for i, old in enumerate(uniq)}
        out["cluster_id"] = out["cluster_id"].map(remap).astype(int)
    else:
        # remap only non-noise
        mask = out["cluster_id"] >= 0
        uniq = sorted(out.loc[mask, "cluster_id"].unique().tolist())
        remap = {old: i for i, old in enumerate(uniq)}
        out.loc[mask, "cluster_id"] = out.loc[mask, "cluster_id"].map(remap).astype(int)

    return out


# =========================
# Optional HDBSCAN recursion
# =========================

@dataclass
class _HNode:
    path: Tuple[int, ...]
    idx: np.ndarray
    depth: int

def recursive_refine_hdbscan(
    df: pd.DataFrame,
    *,
    embedding_col: str = "embedding",
    base_cluster_col: str = "cluster_id",
    max_depth: int = 0,
    min_leaf_size: int = 20,
    min_new_clusters: int = 2,
    min_cluster_size: int = 8,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    pca_n_components: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Apply HDBSCAN recursively to large clusters (usually not needed, but available for extra granularity).

    Stops splitting if:
      - depth reached,
      - cluster size < 2 * min_leaf_size,
      - HDBSCAN returns < min_new_clusters clusters.
    """
    if max_depth <= 0 or df.empty:
        out = df.copy()
        out["hcluster_path"] = out[base_cluster_col].astype(str)
        out["hcluster_id"] = out[base_cluster_col].astype(int)
        return out

    rows: List[Tuple[int, str]] = []

    for top_id, grp in df.groupby(base_cluster_col):
        idx_all = grp.index.to_numpy()
        root = _HNode(path=(int(top_id),), idx=idx_all, depth=0)

        stack: List[_HNode] = [root]
        leaves: List[_HNode] = []

        while stack:
            node = stack.pop()

            # Need capacity for at least two viable children
            if node.depth >= max_depth or len(node.idx) < 2 * min_leaf_size:
                leaves.append(node)
                continue

            sub = df.loc[node.idx].copy()

            # Re-cluster this subset with HDBSCAN
            sub_clustered = cluster_embeddings_hdbscan(
                sub,
                embedding_col=embedding_col,
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric=metric,
                pca_n_components=pca_n_components,
            )

            labels = sub_clustered["cluster_id"].values
            uniq = np.unique(labels)

            if len(uniq) < min_new_clusters:
                leaves.append(node)
                continue

            # Split into viable children
            for cid in uniq:
                child_idx = sub_clustered.index[sub_clustered["cluster_id"] == cid].to_numpy()
                if len(child_idx) < min_leaf_size:
                    continue
                child = _HNode(
                    path=node.path + (int(cid),),
                    idx=child_idx,
                    depth=node.depth + 1,
                )
                stack.append(child)

        # record leaf assignments
        for leaf in leaves:
            pstr = "/".join(map(str, leaf.path))
            for i in leaf.idx:
                rows.append((int(i), pstr))

    assign = pd.DataFrame(rows, columns=["_idx", "hcluster_path"]).set_index("_idx")
    out = df.copy().join(assign, how="left")
    uniq_paths = sorted(out["hcluster_path"].dropna().unique().tolist())
    path2id = {p: j for j, p in enumerate(uniq_paths)}
    out["hcluster_id"] = out["hcluster_path"].map(path2id).astype(int)

    if verbose:
        print(f"[HDBSCAN refine] produced {len(uniq_paths)} leaf paths")

    return out


# ==================
# Cluster centroids
# ==================

def cluster_centroids(df_clusters: pd.DataFrame, embedding_col: str = "embedding") -> pd.DataFrame:
    """
    Compute centroids for each cluster (mean in embedding space).

    Returns columns:
      - cluster_id (int)
      - size (int)
      - centroid (np.ndarray)
    """
    if df_clusters.empty:
        return pd.DataFrame(columns=["cluster_id", "size", "centroid"])

    rows = []
    for cid, grp in df_clusters.groupby("cluster_id"):
        X = np.vstack(grp[embedding_col].to_list())
        centroid = X.mean(axis=0)
        rows.append({"cluster_id": int(cid), "size": len(grp), "centroid": centroid})
    return pd.DataFrame(rows)


# =======================================
# Compatibility shims (keep old imports!)
# =======================================

def cluster_embeddings(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
    target_cluster_size: int = 200,   # ignored for HDBSCAN (kept for signature comp.)
    random_state: int = 42,           # ignored for HDBSCAN
) -> pd.DataFrame:
    """
    Backward-compatible shim to keep `from pysemdist.cluster import cluster_embeddings` working.
    Uses HDBSCAN under the hood with reasonable defaults.
    """
    return cluster_embeddings_hdbscan(
        df,
        embedding_col=embedding_col,
        min_cluster_size=8,
        min_samples=None,
        metric="euclidean",
        pca_n_components=None,
    )


def recursive_refine_clusters(
    df: pd.DataFrame,
    *,
    embedding_col: str = "embedding",
    base_cluster_col: str = "cluster_id",
    max_depth: int = 0,
    min_leaf_size: int = 20,
    min_rel_improvement: float = 0.02,  # not used by HDBSCAN refine; kept for signature compat.
    target_cluster_size: int = 200,     # not used by HDBSCAN refine
    random_state: int = 42,             # not used by HDBSCAN refine
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Backward-compatible shim to keep `from pysemdist.cluster import recursive_refine_clusters` working.
    Calls HDBSCAN-based refinement; ignores KMeans-specific knobs.
    """
    return recursive_refine_hdbscan(
        df,
        embedding_col=embedding_col,
        base_cluster_col=base_cluster_col,
        max_depth=max_depth,
        min_leaf_size=min_leaf_size,
        min_new_clusters=2,
        min_cluster_size=8,
        min_samples=None,
        metric="euclidean",
        pca_n_components=None,
        verbose=verbose,
    )
