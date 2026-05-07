from google import genai
from google.genai import types

from app.config import GEMINI_API_KEY, GEMINI_EMBEDDING_MODEL

_client = None


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
    correct task type is set on each side.
    """
    client = _get_client()
    result = client.models.embed_content(
        model=GEMINI_EMBEDDING_MODEL,
        contents=texts,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return [embedding.values for embedding in result.embeddings]
