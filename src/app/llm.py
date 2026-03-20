import json
import os
import re
from collections.abc import AsyncGenerator

import httpx

from app.classifier import classify_query
from app.config import CLASSIFIER_ENABLED, GENERATION_MODEL, SYSTEM_PROMPT

OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
GENERATION_TIMEOUT = 120.0

_JAILBREAK_PATTERNS = [
    re.compile(p, re.IGNORECASE)
    for p in [
        # Prompt override attempts
        r"forget\s+(all\s+)?(previous|prior|your|the)\s+(instructions|context|rules|prompt)",
        r"ignore\s+(all\s+)?(previous|prior|your|the)\s+(instructions|context|rules|prompt)",
        r"disregard\s+(all\s+)?(previous|prior|your|the)\s+(instructions|context|rules|prompt)",
        r"override\s+(all\s+)?(previous|prior|your|the)\s+(instructions|context|rules|prompt)",
        # Identity reassignment (with role/persona keywords to avoid false positives)
        r"you\s+are\s+(now|no\s+longer)\s+(a|an|my|free|unrestricted|unfiltered|bound\s+by)\b",
        # Persona adoption
        r"act\s+as\s+if\s+you\s+(are|were)\b",
        r"(?:^|[.!?;]\s*)(?:(?:please|now)\s+)?act\s+as\s+(a|an)\s+",
        r"pretend\s+(you\s+are|to\s+be)\s+",
        r"(wake\s+up|break\s+free|come\s+free|set\s+you\s+free)",
        r"you'?re\s+in\s+a\s+(dream|simulation)",
        # Persona switching
        r"(new|switch|change)\s+(persona|personality|identity)\b",
        r"(switch|change)\s+your\s+role\b",
        r"do\s+not\s+follow\s+(your|the)\s+(instructions|rules|prompt)",
        r"jailbreak",
        r"DAN\s+mode",
    ]
]

REFUSAL_MESSAGE = "I can only assist with questions about the provided content."


def _is_jailbreak(query: str) -> bool:
    """Return True if the query matches known jailbreak patterns."""
    return any(pattern.search(query) for pattern in _JAILBREAK_PATTERNS)


def _build_context_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = (
            f"[{i}] {chunk['section_id']} — "
            f"{chunk['title']}. ({chunk['subpart']})"
        )
        parts.append(f"{header}\n{chunk['text']}")
    return "\n\n".join(parts)


async def generate_rag_answer(
    query: str,
    context_chunks: list[dict],
    model: str = GENERATION_MODEL,
) -> str:
    # --- Adversary detection ---
    if _is_jailbreak(query):
        return REFUSAL_MESSAGE

    if CLASSIFIER_ENABLED:
        result = await classify_query(query)
        if not result.passed:
            return REFUSAL_MESSAGE

    # --- Build prompt ---
    context_text = _build_context_prompt(context_chunks)
    system_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}"
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    # --- Call Ollama chat API (non-streaming) ---
    async with httpx.AsyncClient(
        base_url=OLLAMA_BASE_URL, timeout=GENERATION_TIMEOUT
    ) as client:
        resp = await client.post(
            "/api/chat",
            json={"model": model, "messages": messages, "stream": False},
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]


async def generate_rag_answer_stream(
    query: str,
    context_chunks: list[dict],
    model: str | None = None,
) -> AsyncGenerator[str, None]:
    # --- Adversary detection ---
    if _is_jailbreak(query):
        yield REFUSAL_MESSAGE
        return

    if CLASSIFIER_ENABLED:
        result = await classify_query(query)
        if not result.passed:
            yield REFUSAL_MESSAGE
            return

    resolved_model = model or GENERATION_MODEL
    context_text = _build_context_prompt(context_chunks)
    system_content = f"{SYSTEM_PROMPT}\n\nContext:\n{context_text}"

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": query},
    ]

    payload = {
        "model": resolved_model,
        "messages": messages,
        "stream": True,
    }

    # --- Stream from Ollama chat API ---
    async with httpx.AsyncClient(
        base_url=OLLAMA_BASE_URL, timeout=GENERATION_TIMEOUT
    ) as client:
        async with client.stream("POST", "/api/chat", json=payload) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("done"):
                    return
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content
