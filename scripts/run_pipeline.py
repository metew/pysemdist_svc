#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

# Project imports
from pysemdist.config import Config
from pysemdist.io_utils import load_petitions, ensure_dirs
from pysemdist.preprocess import preprocess_df
from pysemdist.embed import embed_texts
from pysemdist.cluster import (
    cluster_centroids,
    recursive_refine_clusters,   # HDBSCAN-based shim in your cluster.py
)
# We’ll call the explicit HDBSCAN entrypoint to control params
from pysemdist.cluster import cluster_embeddings_hdbscan

from pysemdist.label import label_cluster
from pysemdist.meta import meta_cluster
from pysemdist.llm import OpenAICompatClient
from pysemdist.summarize import LLMSummarizer  # emits up to 3 tasks (not forced to 3)

# Optional DB ingestion
from sqlalchemy import create_engine, text as sa_text


def parse_args():
    ap = argparse.ArgumentParser(description="End-to-end petition clustering & summarization")

    # Input sources (choose one): inline sample, DB, CSV, or JSON/JSONL
    ap.add_argument("--sample_inline", action="store_true",
                    help="Use a hardcoded pandas DataFrame sample instead of any file/DB")

    # DB options
    ap.add_argument("--db_url", help="SQLAlchemy URL, e.g., postgresql+psycopg2://user:pass@host:5432/dbname")
    ap.add_argument("--db_query", help="Explicit SQL query returning columns: id, text, optional category, locale")
    ap.add_argument("--db_table", help="Table name if not using --db_query")
    ap.add_argument("--db_where", default="", help="Optional WHERE clause without the word WHERE")

    # CSV / JSON options
    ap.add_argument("--csv_path", help="Path to a CSV with columns id,text,[category],[locale]")
    ap.add_argument("--input", help="Input .json or .jsonl (id, text, optional category). Ignored if other inputs are provided.")

    # Outputs and modeling
    ap.add_argument("--outdir", required=True, help="Output directory for parquet artifacts")
    ap.add_argument("--model", default="intfloat/e5-base-v2", help="SentenceTransformer model for embeddings")
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--target_cluster_size", type=int, default=200)
    ap.add_argument("--n_meta", type=int, default=40)
    ap.add_argument("--categories", nargs="*", default=None, help="Subset of categories to process incrementally")

    # Optional PCA denoise before clustering (general, not HDBSCAN-specific)
    ap.add_argument("--pca_n_components", type=int, default=None,
                    help="Max dimensions to use")

    # Recursive sub-clustering (reuses clustering routine; with HDBSCAN usually not needed)
    ap.add_argument("--subcluster_depth", type=int, default=0, help="Recursive depth (0 disables)")
    ap.add_argument("--sub_min_size", type=int, default=30, help="Minimum leaf size")
    ap.add_argument("--sub_min_improvement", type=float, default=0.02,
                    help="(kept for compatibility; HDBSCAN refine ignores this)")
    ap.add_argument("--sub_target_cluster_size", type=int, default=None,
                    help="(kept for compatibility; HDBSCAN refine ignores this)")
    
    # HDBSCAN knobs
    ap.add_argument("--hdb_metric", default="euclidean",
                    choices=["euclidean", "cosine"],
                    help="Distance metric for HDBSCAN. For cosine on small datasets, we use precomputed distances.")
    ap.add_argument("--hdb_min_cluster_size", type=int, default=8,
                    help="Minimum cluster size for HDBSCAN.")
    ap.add_argument("--hdb_min_samples", type=int, default=None,
                    help="Minimum samples for HDBSCAN (None => equals min_cluster_size).")
    ap.add_argument("--hdb_selection", default="eom",
                    choices=["eom", "leaf"],
                    help="Cluster selection method: 'eom' (broad/stable) or 'leaf' (fine-grained).")
    ap.add_argument("--hdb_epsilon", type=float, default=0.0,
                    help="Optional epsilon threshold when selecting clusters (advanced).")

    # LLM config (OpenAI-compatible / Ollama)
    ap.add_argument("--llm_base_url", default="http://localhost:11434/v1",
                    help="OpenAI-compatible base URL (Ollama, vLLM, llama.cpp w/ compat routes)")
    ap.add_argument("--llm_model", default="yi:9b-chat",
                    help="Model name served by your endpoint (e.g., Yi-9B-Chat, yi:9b-chat)")
    ap.add_argument("--llm_api_key", default="sk-local", help="API key if your server requires one")
    ap.add_argument("--max_exemplars", type=int, default=20, help="Max exemplar texts per cluster for summarization")
    ap.add_argument("--llm_options",
                    type=str,
                    default='{"temperature":0.0,"repeat_penalty":1.05,"num_thread":6,"num_batch":512,"num_ctx":1024,"keep_alive":-1}',
                    help="JSON string of runtime options to include in /chat/completions payload.")

    # Summarization filtering
    ap.add_argument("--summarize_min_size", type=int, default=2,
                    help="Only summarize clusters with at least this many items.")
    ap.add_argument("--summarize_skip_noise", action="store_true", default=True,
                    help="Skip HDBSCAN noise (-1) clusters when summarizing.")

    return ap.parse_args()


def load_from_db(db_url: str, db_query: str | None, db_table: str | None, db_where: str | None) -> pd.DataFrame:
    engine = create_engine(db_url)
    if db_query:
        df = pd.read_sql_query(sa_text(db_query), engine)
    else:
        if not db_table:
            raise SystemExit("Provide --db_table or --db_query with --db_url")
        where = f" WHERE {db_where}" if db_where else ""
        query = f"SELECT id, text, category, locale FROM {db_table}{where}"
        df = pd.read_sql_query(sa_text(query), engine)

    if "id" not in df.columns or "text" not in df.columns:
        raise SystemExit("DB result must include columns: id, text")
    for col in ["category", "locale"]:
        if col not in df.columns:
            df[col] = None
    return df


def load_from_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" not in df.columns or "text" not in df.columns:
        raise SystemExit("CSV must include columns: id, text")
    for col in ["category", "locale"]:
        if col not in df.columns:
            df[col] = None
    return df


def load_inline_sample() -> pd.DataFrame:
    data = [
        {"category": "transport",
         "text": "The petition urges better public transport options in the suburban area. Buses are infrequent and always crowded.",
         "id": "1"},
        {"category": "water",
         "text": "This petition is about recurring water outages in the southern sector. Clean water access is a basic necessity and must be guaranteed.",
         "id": "2"},
        {"category": "hockey", "text": "This petition is about jason wants to own an AHL hockey team.", "id": "3"},
        {"category": "hockey", "text": "jason needs a hockey team, someone get him a hockey team he likes AHL.", "id": "4"},
        {"category": "hockey", "text": "we dont want ahl hockey in our town", "id": "5"},
    ]
    return pd.DataFrame(data, columns=["id", "text", "category"])


def main():
    args = parse_args()

    cfg = Config(
        outdir=Path(args.outdir),
        model_name=args.model,
        batch_size=args.batch_size,
        target_cluster_size=args.target_cluster_size,
        n_meta=args.n_meta,
    )

    ensure_dirs([cfg.silver_embeddings, cfg.gold_clusters, cfg.gold_cluster_summaries, cfg.gold_meta_clusters])

    # Input precedence
    if args.sample_inline:
        df = load_inline_sample()
    elif args.db_url:
        df = load_from_db(args.db_url, args.db_query, args.db_table, args.db_where)
    elif args.csv_path:
        df = load_from_csv(args.csv_path)
    else:
        if not args.input:
            raise SystemExit("Provide --sample_inline, or --db_url, or --csv_path, or --input JSON/JSONL")
        df = load_petitions(args.input)

    if args.categories:
        df = df[df["category"].isin(args.categories)].copy()
        if df.empty:
            print("No rows for requested categories.")
            return 0

    # Preprocess
    df = preprocess_df(df)
    df = df.drop_duplicates(subset=["hash"]).reset_index(drop=True)

    # Embeddings
    print(f"Embedding {len(df)} petitions with model {cfg.model_name}...")
    emb = embed_texts(df["text_norm"].tolist(), model_name=cfg.model_name, batch_size=cfg.batch_size)
    df["embedding"] = [e.astype(np.float32) for e in emb]

    # -------------------------
    # Clustering (HDBSCAN)
    # -------------------------
    print("Clustering...")
    dfc = cluster_embeddings_hdbscan(
        df,
        embedding_col="embedding",
        min_cluster_size=args.hdb_min_cluster_size,
        min_samples=args.hdb_min_samples,
        metric=args.hdb_metric,
        cluster_selection_method=args.hdb_selection,
        cluster_selection_epsilon=args.hdb_epsilon,
        pca_n_components=args.pca_n_components,                              # 64,128,None
        allow_singleton_noise=False,   # ← key change
    )

    # Optional recursive refinement (HDBSCAN-based shim; usually not needed)
    if args.subcluster_depth > 0:
        print(f"Refining with recursive sub-clustering: depth={args.subcluster_depth}, min_size={args.sub_min_size}")
        dfc = recursive_refine_clusters(
            dfc,
            embedding_col="embedding",
            base_cluster_col="cluster_id",
            max_depth=args.subcluster_depth,
            min_leaf_size=args.sub_min_size,
            # other args ignored by HDBSCAN refine shim
        )
        dfc["final_cluster_id"] = dfc["hcluster_id"].astype(int)
    else:
        dfc["final_cluster_id"] = dfc["cluster_id"].astype(int)

    # Persist clusters parquet (+ local sample CSV next to parquet)
    dfc_out = dfc[["id", "category", "lang", "text_norm", "final_cluster_id", "cluster_score", "embedding"]].copy()
    dfc_out = dfc_out.rename(columns={"final_cluster_id": "cluster_id"})
    clusters_path = cfg.gold_clusters
    dfc_out.to_parquet(clusters_path, index=False)

    clusters_sample_csv = clusters_path.parent / "clusters_sample.csv"
    dfc_out.head(50).drop(columns=["embedding"], errors="ignore").assign(embedding=None).to_csv(
        clusters_sample_csv, index=False
    )
    print(f"Wrote clusters → {clusters_path}")


    # -------------------------
    # Cluster membership mapping (lean)
    # -------------------------
    # Prefer the final/leaf cluster id if present, else use cluster_id
    cluster_col = "final_cluster_id" if "final_cluster_id" in dfc.columns else "cluster_id"

    members_cols = [
        "id",                          # petition id
        cluster_col,                   # final cluster id (or base)
        "cluster_score",               # probability (HDBSCAN) or -distance (k-means)
    ]
    # include optional hierarchical path if you have it
    if "hcluster_path" in dfc.columns:
        members_cols.append("hcluster_path")

    members = dfc[members_cols].copy()
    members = members.rename(columns={cluster_col: "cluster_id", "id": "petition_id"})

    # rank by "centrality" within cluster (higher score = more central)
    members["rank_in_cluster"] = (
        members.groupby("cluster_id")["cluster_score"]
            .rank(method="first", ascending=False)
            .astype(int)
    )

    # mark noise if you keep -1 as noise with HDBSCAN
    members["is_noise"] = (members["cluster_id"] == -1)

    # write it
    members_path = clusters_path.parent / "cluster_members.parquet"
    members.to_parquet(members_path, index=False)

    # handy small CSV sample (human-friendly)
    (members
    .sort_values(["cluster_id", "rank_in_cluster"])
    .head(200)
    .to_csv(clusters_path.parent / "cluster_members_sample.csv", index=False))

    print(f"Wrote cluster members → {members_path}")


    # -------------------------
    # Cluster summary stats
    # -------------------------
    print("Generating cluster summary statistics...")
    counts = dfc_out["cluster_id"].value_counts().sort_index()
    stats = dfc_out.groupby("cluster_id")["cluster_score"].agg(["count", "mean", "min", "max"])
    summary_lines = [
        f"{len(dfc_out)} petitions",
        f"{dfc_out['cluster_id'].nunique()} clusters",
        "",
        str(stats),
    ]
    summary_str = "\n".join(summary_lines)
    print(summary_str)

    summary_file = clusters_path.parent / "clusters_stats.txt"
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(summary_str)
    print(f"Wrote cluster summary stats → {summary_file}")


    # --- LLM-backed summarizer ---
    client = OpenAICompatClient(base_url=args.llm_base_url, api_key=args.llm_api_key, model=args.llm_model)
    client.options = args.llm_options  # attach JSON string with Ollama runtime opts
    summarizer = LLMSummarizer(client=client)

    def _normalize_tasks(x):
        """Return a list (max 3) of tasks, robust to weird LLM outputs."""
        if x is None:
            return []
        if isinstance(x, str):
            # split on newlines/bullets, strip empties
            parts = [t.strip(" -•\t") for t in x.splitlines() if t.strip()]
            return parts[:3]
        if isinstance(x, (list, tuple)):
            parts = []
            for t in x:
                if t is None:
                    continue
                if isinstance(t, str):
                    t = t.strip()
                    if t:
                        parts.append(t)
            return parts[:3]
        return []

    print("Labeling & summarizing clusters with LLM...")

    # Decide how to treat noise/singletons:
    AGGREGATE_NOISE = True   # put all noise into one “Misc/Noise” summary
    SUMMARIZE_SINGLETONS = True  # let singletons get their own problem/tasks

    rows = []

    # Identify noise if present (HDBSCAN with allow_singleton_noise=False keeps -1)
    cluster_col = "final_cluster_id" if "final_cluster_id" in dfc.columns else "cluster_id"
    grouped = list(dfc.groupby(cluster_col))

    # Optional: aggregate noise into a single bucket
    if AGGREGATE_NOISE and any(cid == -1 for cid, _ in grouped):
        noise_grp = dfc[dfc[cluster_col] == -1]
        if not noise_grp.empty:
            label = "Noise / Mixed"
            keywords = []
            exemplars = noise_grp.sort_values("cluster_score", ascending=False)\
                                .head(args.max_exemplars)["text_norm"].tolist()
            summary = summarizer.problem_and_tasks(exemplars)
            tasks = _normalize_tasks(summary.get("decision_tasks"))
            rows.append({
                "cluster_id": int(-1),
                "size": int(len(noise_grp)),
                "keywords": keywords,
                "label": label,
                "problem_statement": summary.get("problem_statement", "").strip(),
                "decision_tasks": tasks,
                "exemplars": noise_grp.sort_values("cluster_score", ascending=False).head(10)["id"].astype(str).tolist(),
            })

    # Summarize all non-noise clusters (and optionally singletons)
    from tqdm import tqdm
    for cid, grp in tqdm((g for g in grouped if g[0] != -1), total=sum(1 for g in grouped if g[0] != -1)):
        n = len(grp)
        if n < 2 and not SUMMARIZE_SINGLETONS:
            continue

        label, keywords = label_cluster(grp)
        exemplars = grp.sort_values("cluster_score", ascending=False)\
                    .head(args.max_exemplars)["text_norm"].tolist()
        summary = summarizer.problem_and_tasks(exemplars)
        tasks = _normalize_tasks(summary.get("decision_tasks"))

        rows.append({
            "cluster_id": int(cid),
            "size": int(n),
            "keywords": keywords,
            "label": label,
            "problem_statement": summary.get("problem_statement", "").strip(),
            "decision_tasks": tasks,
            "exemplars": grp.sort_values("cluster_score", ascending=False).head(10)["id"].astype(str).tolist(),
        })

    # Build df_summary safely
    expected_cols = ["cluster_id", "size", "keywords", "label", "problem_statement", "decision_tasks", "exemplars"]
    if rows:
        df_summary = pd.DataFrame(rows)
        # Ensure all expected columns exist
        for c in expected_cols:
            if c not in df_summary.columns:
                df_summary[c] = [] if c in ("keywords", "decision_tasks", "exemplars") else ""
    else:
        df_summary = pd.DataFrame(columns=expected_cols)

    # Final safety normalization (in case anything weird slipped through)
    df_summary["decision_tasks"] = df_summary["decision_tasks"].apply(_normalize_tasks)

    
    df_summary_path = cfg.gold_cluster_summaries
    df_summary.to_parquet(df_summary_path, index=False)

    cluster_summaries_sample_csv = df_summary_path.parent / "cluster_summaries_sample.csv"
    df_summary.head(50).to_csv(cluster_summaries_sample_csv, index=False)
    print(f"Wrote cluster summaries → {df_summary_path}")


    # -------------------------
    # Meta clustering on centroids of final clusters
    # -------------------------
    # Use exactly one 'cluster_id' column = the final leaf id for centroids
    dfc_for_meta = dfc_out.copy()

    cents = cluster_centroids(dfc_for_meta, embedding_col="embedding")
    df_meta_map = meta_cluster(cents, n_meta=cfg.n_meta)

    if not df_meta_map.empty:
        merged = df_meta_map.merge(
            df_summary[["cluster_id", "label", "problem_statement", "decision_tasks"]],
            on="cluster_id",
            how="left",
        )
        theme_rows = []
        for mid, grp in merged.groupby("meta_id"):
            labels = " ".join(grp["label"].astype(str).tolist()).split()
            top = [w for w, _ in Counter(labels).most_common(3)]
            theme_label = " ".join(top) or "General Theme"
            ps = max(grp["problem_statement"].astype(str).tolist(), key=len, default="Related public issues.")
            tasks = []
            for ts in grp["decision_tasks"].tolist():
                if isinstance(ts, (list, tuple)):
                    for t in ts:
                        if isinstance(t, str) and t not in tasks:
                            tasks.append(t)
            
            theme_rows.append(
                {
                    "meta_id": int(mid),
                    "theme_label": theme_label,
                    "meta_problem_statement": ps,
                    "canonical_tasks": tasks,
                    "included_cluster_ids": sorted(grp["cluster_id"].astype(int).tolist()),
                }
            )
        df_meta = pd.DataFrame(theme_rows)
    else:
        df_meta = pd.DataFrame(
            columns=["meta_id", "theme_label", "meta_problem_statement", "canonical_tasks", "included_cluster_ids"]
        )

    df_meta_path = cfg.gold_meta_clusters
    df_meta.to_parquet(df_meta_path, index=False)

    meta_clusters_sample_csv = df_meta_path.parent / "meta_clusters_sample.csv"
    df_meta.head(50).to_csv(meta_clusters_sample_csv, index=False)
    print(f"Wrote meta clusters → {df_meta_path}")

    # Compact preview
    preview = {
        "clusters": dfc_out[["id", "cluster_id", "cluster_score"]].head(3).to_dict(orient="records"),
        "cluster_summaries": df_summary.head(3).to_dict(orient="records"),
        "meta_clusters": df_meta.head(5).to_dict(orient="records"),
    }
    print(json.dumps(preview, indent=2))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
