import os
from datetime import datetime
from typing import Dict
import pandas as pd
from .config import settings

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def save_results(df: pd.DataFrame, group: str) -> Dict[str, str]:
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    base_dir = os.path.join(settings.output_dir, group)
    ensure_dir(base_dir)
    csv_path = os.path.join(base_dir, f"{group}_{ts}.csv")
    df.to_csv(csv_path, index=False)
    return {"csv": csv_path}

def maybe_upload_s3(paths: Dict[str, str]) -> Dict[str, str]:
    if not settings.s3_bucket or not settings.s3_upload_enabled:
        # Still return planned destinations for visibility
        if settings.s3_bucket:
            return {k: f"s3://{settings.s3_bucket}/{settings.s3_prefix}/{os.path.basename(v)}" for k, v in paths.items()}
        return {}
    # Real upload
    import boto3
    s3 = boto3.client("s3")
    uploaded = {}
    for key, local in paths.items():
        dest_key = f"{settings.s3_prefix}/{os.path.basename(local)}".lstrip("/")
        s3.upload_file(local, settings.s3_bucket, dest_key)
        uploaded[key] = f"s3://{settings.s3_bucket}/{dest_key}"
    return uploaded
