from __future__ import annotations
from typing import List, Dict
from .llm import OpenAICompatClient


SYSTEM_PROMPT = (
    "You are a policy analyst. You convert clusters of citizen petitions into neutral, generic problem statements "
    "and concise, non-partisan decision-maker tasks.\n"
    "Rules:\n"
    "1. Avoid proper nouns and specific locations unless explicitly required.\n"
    "2. Abstract to the underlying service or governance problem.\n"
    "3. Be factual and even-toned.\n"
    "4. Produce actions framed as imperative verbs.\n"
    "5. Keep the output schema exactly as instructed."
)

def PROBLEM_TEMPLATE(snippets: str) -> str:
    return f"""
Summarize the shared public problem described across these petition snippets as a neutral, generic statement suitable for policymakers.

Snippets (deduplicated, trimmed):
{snippets}

Output ONLY a single paragraph (2 sentences max). Avoid names, places, and campaign-specific asks. Focus on the underlying problem class (e.g., reliability, access, safety, equity, accountability).
"""

def TASKS_TEMPLATE(problem: str) -> str:
    return f"""
Based on the problem below, propose up to 3 concise, generic actions a responsible authority could take.
- Use imperative verbs.
- Non-partisan.
- Avoid proper nouns.
- No more than 20 words each.
- If only 1 or 2 are clearly sufficient, return only those.

Problem:
{problem}
"""


class LLMSummarizer:
    """LLM-backed generic summarizer using an OpenAI-compatible chat endpoint (vLLM, llama.cpp, Ollama, etc.)."""

    def __init__(
        self,
        client: OpenAICompatClient,
        max_exemplar_chars: int = 2400,
        max_snippets: int = 30,
        temperature: float = 0.2,
    ) -> None:
        self.client = client
        self.max_exemplar_chars = max_exemplar_chars
        self.max_snippets = max_snippets
        self.temperature = temperature

    def _truncate_snippets(self, texts: List[str]) -> List[str]:
        """Keep only enough snippets to stay under character limit."""
        out, total = [], 0
        for t in texts[: self.max_snippets]:
            t = t.strip()
            if not t:
                continue
            if total + len(t) + 1 > self.max_exemplar_chars:
                break
            out.append(t)
            total += len(t) + 1
        return out

    def _chat(self, prompt: str) -> str:
        """Send a single prompt to the chat model."""
        return self.client.chat(
            system=SYSTEM_PROMPT,
            user=prompt,
            temperature=self.temperature,
            max_tokens=256,
        )

    def problem_and_tasks(self, exemplar_texts: List[str]) -> Dict[str, List[str] | str]:
        snippets = self._truncate_snippets(exemplar_texts)
        joined = "\n- " + "\n- ".join(snippets) if snippets else "(no snippets)"

        problem = self._chat(PROBLEM_TEMPLATE(joined)).strip()

        tasks_raw = self._chat(TASKS_TEMPLATE(problem)).strip()
        tasks = [t.strip("-• \t") for t in tasks_raw.split("\n") if t.strip()]
        tasks = tasks[:3]  # cap at 3; allow 1–3
        return {"problem_statement": problem, "decision_tasks": tasks}
