from __future__ import annotations
import math
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import MiniBatchKMeans

def compute_k(n: int, target_cluster_size: int) -> int:
    """Compute the number of clusters k based on dataset size and target cluster size."""
    return max(2, math.ceil(n / max(20, target_cluster_size)))

def cluster_embeddings(
    df: pd.DataFrame,
    embedding_col: str = "embedding",
    target_cluster_size: int = 200,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Cluster embeddings using MiniBatchKMeans.

    Returns:
        A copy of df with two new columns:
        - cluster_id: integer cluster assignment
        - cluster_score: negative distance to centroid (higher = closer)
    """
    if df.empty:
        return df.assign(cluster_id=[], cluster_score=[])

    X = np.vstack(df[embedding_col].to_list())

    # Dimensionality reduction for stability/speed.
    # Cap n_components by both feature count and (n_samples - 1).
    # Skip PCA for very small batches to avoid errors like:
    # ValueError: n_components must be between 0 and min(n_samples, n_features)
    n_samples, n_features = X.shape
    n_comp = min(128, n_features, max(1, n_samples - 1))

    if n_comp >= 2 and n_comp < n_features:
        # randomized solver is faster for larger matrices
        pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=random_state)
        Xp = pca.fit_transform(X)
    else:
        # when batch is tiny or reduction not meaningful, skip PCA
        Xp = X

    k = compute_k(len(df), target_cluster_size)
    kmeans = MiniBatchKMeans(
        n_clusters=k, random_state=random_state, batch_size=2048
    )
    labels = kmeans.fit_predict(Xp)
    centers = kmeans.cluster_centers_

    # Compute distances to centroid
    from numpy.linalg import norm
    dists = np.array([norm(x - centers[l]) for x, l in zip(Xp, labels)])

    out = df.copy()
    out["cluster_id"] = labels.astype(int)
    out["cluster_score"] = -dists  # higher is better (closer)
    return out

def cluster_centroids(
    df_clusters: pd.DataFrame, embedding_col: str = "embedding"
) -> pd.DataFrame:
    """Compute centroids for each cluster."""
    rows = []
    for cid, grp in df_clusters.groupby("cluster_id"):
        X = np.vstack(grp[embedding_col].to_list())
        centroid = X.mean(axis=0)
        rows.append({"cluster_id": int(cid), "size": len(grp), "centroid": centroid})
    return pd.DataFrame(rows)

# --- Recursive sub-clustering reusing the current clustering process ---

from dataclasses import dataclass
from typing import Tuple, List, Dict

def _sse_inertia(X: np.ndarray) -> float:
    """Sum of squared distances to the mean (like inertia)."""
    if X.size == 0:
        return 0.0
    c = X.mean(axis=0, keepdims=True)
    diffs = X - c
    return float((diffs * diffs).sum())

@dataclass
class _RNode:
    path: Tuple[int, ...]    # e.g., (top_cluster_id, 0, 1)
    idx: np.ndarray          # integer indices into original df
    depth: int
    parent_sse: float        # SSE of this node (before splitting)

def recursive_refine_clusters(
    df: pd.DataFrame,
    *,
    embedding_col: str = "embedding",
    base_cluster_col: str = "cluster_id",
    max_depth: int = 0,
    min_leaf_size: int = 30,
    min_rel_improvement: float = 0.02,  # require e.g. 2% SSE drop to accept a split
    target_cluster_size: int = 200,     # used to compute k at each level
    random_state: int = 42,
    verbose: bool = True,               # NEW: print a one-line summary of split decisions
) -> pd.DataFrame:
    """
    Recursively re-cluster each (sub)cluster using the *same* cluster_embeddings routine.
    Stops splitting a node if:
      - max_depth reached, or
      - any child would be < min_leaf_size, or
      - relative SSE improvement < min_rel_improvement, or
      - the split collapses to a single label.

    Produces:
      - hcluster_path: "top/child/..." (string)
      - hcluster_id: contiguous int per leaf
    """
    if max_depth <= 0 or df.empty:
        out = df.copy()
        out["hcluster_path"] = out[base_cluster_col].astype(str)
        out["hcluster_id"] = out[base_cluster_col].astype(int)
        return out

    # Simple counters to understand why splits were accepted/rejected
    reasons = {"accepted": 0, "depth": 0, "size": 0, "degenerate": 0, "impr": 0}

    rows: List[Tuple[int, str]] = []

    for top_id, grp in df.groupby(base_cluster_col):
        idx_all = grp.index.to_numpy()
        X_all = np.vstack(grp[embedding_col].to_list())
        root = _RNode(path=(int(top_id),), idx=idx_all, depth=0, parent_sse=_sse_inertia(X_all))

        stack: List[_RNode] = [root]
        leaves: List[_RNode] = []

        while stack:
            node = stack.pop()

            # stop by depth or size (need at least 2*min_leaf_size to form two viable children)
            if node.depth >= max_depth or len(node.idx) < 2 * min_leaf_size:
                if node.depth >= max_depth:
                    reasons["depth"] += 1
                else:
                    reasons["size"] += 1
                leaves.append(node)
                continue

            # Run the SAME clustering routine on the node's members
            sub = df.loc[node.idx].copy()
            # compute k for this subset
            k_subset = compute_k(len(sub), target_cluster_size)
            if k_subset < 2:
                reasons["degenerate"] += 1
                leaves.append(node)
                continue

            # IMPORTANT: we don't want to alter global cluster ids inside; we only need new local splits.
            # So we call cluster_embeddings on the subset (returns a frame with a temporary 'cluster_id')
            sub_clustered = cluster_embeddings(
                sub,
                embedding_col=embedding_col,
                target_cluster_size=target_cluster_size,
                random_state=random_state,
            )

            # If clustering produced a single label, no meaningful split
            unique_labels = sub_clustered["cluster_id"].unique()
            if len(unique_labels) < 2:
                reasons["degenerate"] += 1
                leaves.append(node)
                continue

            # Evaluate split quality via SSE improvement
            child_sse = 0.0
            child_parts: List[np.ndarray] = []
            valid = True
            for lab in unique_labels:
                child_idx = sub_clustered.index[sub_clustered["cluster_id"] == lab].to_numpy()
                if len(child_idx) < min_leaf_size:
                    valid = False
                    break
                Xc = np.vstack(df.loc[child_idx, embedding_col].to_list())
                child_sse += _sse_inertia(Xc)
                child_parts.append(child_idx)

            if not valid:
                reasons["size"] += 1
                leaves.append(node)
                continue

            if node.parent_sse <= 0:
                rel_impr = 0.0
            else:
                rel_impr = (node.parent_sse - child_sse) / node.parent_sse

            if rel_impr < min_rel_improvement:
                # Not a meaningful refinement
                reasons["impr"] += 1
                leaves.append(node)
                continue

            # Accept split: push children; assign stable order by label sort
            reasons["accepted"] += 1
            for j, child_idx in enumerate(child_parts):
                Xc = np.vstack(df.loc[child_idx, embedding_col].to_list())
                child_node = _RNode(
                    path=node.path + (int(j),),
                    idx=child_idx,
                    depth=node.depth + 1,
                    parent_sse=_sse_inertia(Xc),
                )
                stack.append(child_node)

        # Collect leaf assignments for this top cluster
        for leaf in leaves:
            path_str = "/".join(map(str, leaf.path))
            for i in leaf.idx:
                rows.append((int(i), path_str))

    # Build output
    assign_df = pd.DataFrame(rows, columns=["_idx", "hcluster_path"]).set_index("_idx")
    out = df.copy()
    out = out.join(assign_df, how="left")
    # Map unique paths to contiguous ints
    uniq = {p: j for j, p in enumerate(sorted(out["hcluster_path"].dropna().unique().tolist()))}
    out["hcluster_id"] = out["hcluster_path"].map(uniq).astype(int)

    if verbose:
        print("Sub-cluster decisions:", reasons)

    return out
