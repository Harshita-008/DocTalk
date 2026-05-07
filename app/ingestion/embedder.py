import re
import time

from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

_client = None

_MAX_ATTEMPTS = 4
_MAX_RETRY_SLEEP = 30


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=GEMINI_API_KEY)
    return _client


def get_embeddings(
    texts: list[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> list[list[float]]:
    """Embed texts using Google's Gemini embeddings API.

    Use task_type="RETRIEVAL_QUERY" when embedding search queries — Gemini
    produces asymmetric document/query embeddings that align better when the
    correct task type is set on each side. Retries on 429s, capping each
    sleep so a single batch can't exhaust the function timeout.
    """
    client = _get_client()

    last_exc = None
    for attempt in range(_MAX_ATTEMPTS):
        try:
            result = client.models.embed_content(
                model=GEMINI_EMBEDDING_MODEL,
                contents=texts,
                config=types.EmbedContentConfig(task_type=task_type),
            )
            return [embedding.values for embedding in result.embeddings]
        except Exception as exc:
            last_exc = exc
            if not _is_rate_limit(exc) or attempt == _MAX_ATTEMPTS - 1:
                raise
            sleep_for = min(_retry_delay(exc) or (2 ** attempt), _MAX_RETRY_SLEEP)
            time.sleep(sleep_for)

    raise last_exc  # unreachable, kept for type checkers


def _is_rate_limit(exc: Exception) -> bool:
    text = str(exc)
    return "429" in text or "RESOURCE_EXHAUSTED" in text


def _retry_delay(exc: Exception) -> int | None:
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s", str(exc))
    return int(match.group(1)) if match else None
