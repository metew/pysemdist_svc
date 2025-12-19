from __future__ import annotations
import re
import hashlib
from dataclasses import dataclass
from typing import List

import pandas as pd
from langdetect import detect

# Regular expressions
WS = re.compile(r"\s+")
URL = re.compile(r"https?://\S+")


def normalize_text(text: str) -> str:
    """Normalize text by lowercasing, stripping whitespace, and removing URLs."""
    text = text or ""
    text = text.strip().lower()
    text = URL.sub("", text)
    text = WS.sub(" ", text)
    return text


def sha1(s: str) -> str:
    """Return SHA1 hash of a string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


@dataclass
class PreprocessStats:
    n: int
    n_empty: int


def preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess a dataframe by normalizing text, hashing, and detecting language."""
    df = df.copy()
    df["text_norm"] = df["text"].astype(str).map(normalize_text)
    df["hash"] = df["text_norm"].map(sha1)

    # Language detection best-effort; skip failures
    langs: List[str] = []
    for t in df["text_norm"].tolist():
        try:
            langs.append(detect(t) if t else "")
        except Exception:
            langs.append("")

    df["lang"] = langs
    return df
