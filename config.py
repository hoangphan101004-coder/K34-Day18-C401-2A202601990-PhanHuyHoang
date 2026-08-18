"""Shared configuration for Lab 18."""

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
_openai_key = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_API_KEY = "" if _openai_key in {"", "sk-...", "sk-your-key-here"} else _openai_key


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# External calls are opt-in so tests and local demos never create surprise cost.
OPENAI_ENRICHMENT_ENABLED = _env_flag("OPENAI_ENRICHMENT_ENABLED", False)
OPENAI_ANSWERS_ENABLED = _env_flag("OPENAI_ANSWERS_ENABLED", False)
RAGAS_ENABLED = _env_flag("RAGAS_ENABLED", False)

# --- Qdrant ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "lab18_production"
NAIVE_COLLECTION = "lab18_naive"

# --- Embedding ---
EMBEDDING_MODEL = "BAAI/bge-m3"
EMBEDDING_DIM = 1024

# --- Chunking ---
HIERARCHICAL_PARENT_SIZE = 2048
HIERARCHICAL_CHILD_SIZE = 256
SEMANTIC_THRESHOLD = 0.85

# --- Search ---
BM25_TOP_K = 20
DENSE_TOP_K = 20
HYBRID_TOP_K = 20
RERANK_TOP_K = 3

# --- Paths ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TEST_SET_PATH = os.path.join(os.path.dirname(__file__), "test_set.json")
