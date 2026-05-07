from openai import OpenAI

from app.config import OPENROUTER_API_KEY, OPENROUTER_EMBEDDING_MODEL

_client = None

# OpenRouter's loop-detection filter can be triggered by repetitive document
# text (e.g. textbook PDFs with repeated headers/footers). Adding this tag to
# the request tells OpenRouter the repetition is intentional.
_LOOP_BYPASS_TAG = "[ignoring loop detection]"


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
        )
    return _client


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Embed texts using the OpenRouter embeddings API.

    Uses NVIDIA Llama Nemotron Embed VL 1B V2 by default — a free,
    high-quality embedding model optimised for retrieval tasks.

    OpenRouter may flag repetitive document text (e.g. repeated PDF headers)
    as "looping content". On that error we retry once with the bypass tag
    prepended so that legitimate document content is never silently dropped.
    """
    client = _get_client()

    try:
        return _call_embeddings(client, texts)
    except Exception as exc:
        # OpenRouter loop-detection error — retry with bypass tag
        if "looping content" in str(exc).lower():
            tagged = [f"{_LOOP_BYPASS_TAG} {t}" for t in texts]
            return _call_embeddings(client, tagged)
        raise


def _call_embeddings(client: OpenAI, texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        model=OPENROUTER_EMBEDDING_MODEL,
        input=texts,
    )
    # Sort by index to guarantee the same order as the input list
    return [item.embedding for item in sorted(response.data, key=lambda x: x.index)]
