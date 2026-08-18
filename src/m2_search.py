from __future__ import annotations

"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    if not text:
        return ""
    try:
        from underthesea import word_tokenize

        segmented = word_tokenize(text, format="text")
    except Exception:
        segmented = text
    return " ".join(segmented.replace("_", " ").lower().split())


class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = list(chunks)
        self.corpus_tokens = [segment_vietnamese(chunk.get("text", "")).split() for chunk in chunks]
        if not self.corpus_tokens:
            self.bm25 = None
            return
        from rank_bm25 import BM25Okapi

        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        """Search using BM25."""
        if self.bm25 is None or top_k <= 0:
            return []
        scores = self.bm25.get_scores(segment_vietnamese(query).split())
        top_indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)
        results = []
        for index in top_indices:
            if float(scores[index]) <= 0:
                continue
            document = self.documents[index]
            results.append(SearchResult(
                text=document.get("text", ""),
                score=float(scores[index]),
                metadata=dict(document.get("metadata", {})),
                method="bm25",
            ))
            if len(results) >= top_k:
                break
        return results


class DenseSearch:
    def __init__(self):
        from qdrant_client import QdrantClient
        self.client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
        self._encoder = None
        self._local_indexes: dict[str, tuple[object, list[dict]]] = {}

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer(EMBEDDING_MODEL)
        return self._encoder

    def collection_ready(self, collection: str = COLLECTION_NAME) -> bool:
        try:
            return self.client.collection_exists(collection) and self.client.count(collection).count > 0
        except Exception:
            return False

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into Qdrant."""
        import numpy as np

        if not chunks:
            self._local_indexes[collection] = (np.empty((0, EMBEDDING_DIM)), [])
            return

        from qdrant_client.models import Distance, PointStruct, VectorParams

        texts = [chunk.get("text", "") for chunk in chunks]
        vectors = self._get_encoder().encode(
            texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True
        )
        vectors = np.asarray(vectors, dtype="float32")
        documents = [
            {"text": chunk.get("text", ""), "metadata": dict(chunk.get("metadata", {}))}
            for chunk in chunks
        ]
        self._local_indexes[collection] = (vectors, documents)

        try:
            if self.client.collection_exists(collection):
                self.client.delete_collection(collection)
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=int(vectors.shape[1]), distance=Distance.COSINE),
            )
            for start in range(0, len(documents), 64):
                points = [
                    PointStruct(
                        id=index,
                        vector=vectors[index].tolist(),
                        payload={**documents[index]["metadata"], "text": documents[index]["text"]},
                    )
                    for index in range(start, min(start + 64, len(documents)))
                ]
                self.client.upsert(collection_name=collection, points=points, wait=True)
        except Exception as exc:
            print(f"  Warning: Qdrant indexing failed ({exc}); using in-memory dense index.")

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        if top_k <= 0:
            return []
        query_vector = self._get_encoder().encode(query, normalize_embeddings=True)
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=query_vector.tolist(),
                limit=top_k,
                with_payload=True,
            )
            results = []
            for point in response.points:
                payload = dict(point.payload or {})
                text = str(payload.pop("text", ""))
                results.append(SearchResult(text, float(point.score), payload, "dense"))
            return results
        except Exception as exc:
            local_index = self._local_indexes.get(collection)
            if local_index is None:
                print(f"  Warning: Qdrant search failed and no local index exists ({exc}).")
                return []
            vectors, documents = local_index
            scores = vectors @ query_vector
            indices = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:top_k]
            return [
                SearchResult(
                    documents[index]["text"], float(scores[index]),
                    dict(documents[index]["metadata"]), "dense",
                )
                for index in indices
            ]


def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    if k < 0:
        raise ValueError("k must be non-negative")
    fused: dict[str, dict] = {}
    for results in results_list:
        for rank, result in enumerate(results):
            entry = fused.setdefault(result.text, {"score": 0.0, "result": result})
            entry["score"] += 1.0 / (k + rank + 1)

    ranked = sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:top_k]
    return [
        SearchResult(
            text=item["result"].text,
            score=float(item["score"]),
            metadata=dict(item["result"].metadata),
            method="hybrid",
        )
        for item in ranked
    ]


class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict], index_dense: bool = True) -> None:
        self.bm25.index(chunks)
        if index_dense:
            self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
