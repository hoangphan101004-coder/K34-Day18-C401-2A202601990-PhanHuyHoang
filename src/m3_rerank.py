from __future__ import annotations

"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    _model_cache: dict[str, object | None] = {}

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            if self.model_name not in self._model_cache:
                try:
                    from sentence_transformers import CrossEncoder

                    self._model_cache[self.model_name] = CrossEncoder(self.model_name)
                except Exception as exc:
                    print(f"  Warning: reranker model unavailable ({exc}); using lexical reranking.")
                    self._model_cache[self.model_name] = None
            self._model = self._model_cache[self.model_name]
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if model is not None:
            try:
                import numpy as np

                pairs = [(query, document.get("text", "")) for document in documents]
                scores = np.asarray(model.predict(pairs, show_progress_bar=False)).reshape(-1)
            except Exception as exc:
                print(f"  Warning: reranker inference failed ({exc}); using lexical reranking.")
                scores = [_lexical_rerank_score(query, document) for document in documents]
        else:
            scores = [_lexical_rerank_score(query, document) for document in documents]

        adjusted_scores = [
            float(score) + _version_adjustment(document)
            for score, document in zip(scores, documents)
        ]
        scored = sorted(
            zip(adjusted_scores, documents), key=lambda item: item[0], reverse=True
        )[:top_k]
        return [
            RerankResult(
                text=document.get("text", ""),
                original_score=float(document.get("score", 0.0)),
                rerank_score=float(score),
                metadata=dict(document.get("metadata", {})),
                rank=rank,
            )
            for rank, (score, document) in enumerate(scored, start=1)
        ]


def _lexical_rerank_score(query: str, document: dict) -> float:
    query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    document_tokens = set(re.findall(r"\w+", document.get("text", "").lower(), flags=re.UNICODE))
    overlap = len(query_tokens & document_tokens) / max(len(query_tokens), 1)
    return overlap + 0.01 * float(document.get("score", 0.0))


def _version_adjustment(document: dict) -> float:
    """Prefer current policy versions when a superseded and current copy coexist."""
    metadata = document.get("metadata", {})
    source = str(metadata.get("source", "")).lower()
    text = document.get("text", "").lower()
    if any(marker in text for marker in ("trạng thái: đã thay thế", "phiên bản cũ")):
        return -3.0
    if any(marker in source for marker in ("_v1.", "_v2023.")):
        return -3.0
    if any(marker in source for marker in ("_v2.", "_v2024.")) or "phiên bản hiện hành" in text:
        return 0.5
    return 0.0


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        # TODO (optional): from flashrank import Ranker, RerankRequest
        # model = Ranker(); passages = [{"text": d["text"]} for d in documents]
        # results = model.rerank(RerankRequest(query=query, passages=passages))
        return []


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs. (Đã implement sẵn)"""
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        elapsed = (time.perf_counter() - start) * 1000
        times.append(elapsed)
    return {"avg_ms": sum(times) / len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
