from __future__ import annotations
import pandas as pd


def compute_exemplars(df_cluster: pd.DataFrame, n: int = 10) -> list[str]:
    return df_cluster.sort_values("cluster_score", ascending=False).head(n)["id"].astype(str).tolist()