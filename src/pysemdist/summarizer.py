from typing import List, Dict
import requests
import os

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "yi:9b")

def summarize_goal(texts: List[str], max_tokens: int = 64) -> str:
    # Build a compact prompt
    excerpt = "\n".join(t[:160] for t in texts[:5])
    prompt = (
        "You are a labeling assistant. Read petition snippets and name the shared goal in 10 words or fewer.\n"
        "Snippets:\n" + excerpt + "\n"
        "Goal:"
    )
    try:
        r = requests.post(f"{OLLAMA_HOST}/api/generate", json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False, "options": {"num_predict": max_tokens}} , timeout=60)
        r.raise_for_status()
        data = r.json()
        return (data.get("response") or "").strip().splitlines()[0]
    except Exception:
        return "Goal label unavailable"
