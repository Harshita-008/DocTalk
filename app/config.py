import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenRouter — used for embeddings (NVIDIA Llama Nemotron Embed VL 1B V2)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_EMBEDDING_MODEL = os.getenv(
    "OPENROUTER_EMBEDDING_MODEL",
    "nvidia/llama-nemotron-embed-vl-1b-v2:free",
)

# ChromaDB Cloud
CHROMADB_API_KEY = os.getenv("CHROMADB_API_KEY")

LLM_MODEL = os.getenv("LLM_MODEL", "google/flan-t5-base")
OPENAI_LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-4o-mini")

# Set to 1 for OpenRouter API — sending chunks one at a time avoids the
# loop-detection filter that fires when a batch contains similar PDF text.
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "1"))
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "10"))
MAX_UPLOAD_BYTES = MAX_UPLOAD_MB * 1024 * 1024

# Chunk size in words. 350 words gives ~430 tokens — enough for coherent
# paragraphs while keeping chunks focused for embedding.
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "70"))

# Retrieve more candidates than we ultimately pass to the model so the
# reranker has enough material to work with.
TOP_K = int(os.getenv("TOP_K", "12"))
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "3"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "3"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "1"))

# Minimum cosine similarity (1 - distance) for a chunk to be considered
# relevant when no keyword signal is present.
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.20"))
