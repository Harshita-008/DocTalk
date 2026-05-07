import hashlib
import math
import re

from app.config import EMBEDDING_MODEL, ENABLE_SENTENCE_TRANSFORMERS


FALLBACK_DIM = 384
_model = None
_model_load_attempted = False


def get_embeddings(texts):
    model = _get_sentence_transformer()
    if model is not None:
        try:
            return model.encode(
                texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()
        except Exception:
            pass

    return [_fallback_embedding(text) for text in texts]


def _get_sentence_transformer():
    global _model, _model_load_attempted

    if not ENABLE_SENTENCE_TRANSFORMERS:
        return None

    if _model is not None:
        return _model
    if _model_load_attempted:
        return None

    _model_load_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        # Allow the model to be downloaded on first use if not already cached.
        # Removing local_files_only=True ensures the embedding model is always
        # available rather than silently falling back to the hash-based stub.
        _model = SentenceTransformer(EMBEDDING_MODEL)
        return _model
    except Exception:
        return None


def _fallback_embedding(text):
    """Hash-based lexical fallback used only when sentence-transformers is
    unavailable.  This has poor semantic recall — it is a last resort.
    """
    tokens = re.findall(r"[a-zA-Z][a-zA-Z-]{2,}", (text or "").lower())
    features = []
    features.extend(tokens)
    features.extend(f"{tokens[i]} {tokens[i + 1]}" for i in range(len(tokens) - 1))

    vector = [0.0] * FALLBACK_DIM
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % FALLBACK_DIM
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [value / norm for value in vector]
