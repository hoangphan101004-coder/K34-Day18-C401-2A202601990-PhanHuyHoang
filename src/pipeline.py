from __future__ import annotations

"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, re, sys, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K


def build_pipeline(rebuild_dense: bool = True):
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE")
    print("=" * 60, flush=True)

    # Step 1: Load & Chunk (M1)
    t0 = time.time()
    print("\n[1/4] Chunking documents...", flush=True)
    docs = load_documents()
    all_chunks = []
    for doc in docs:
        parents, children = chunk_hierarchical(doc["text"], metadata=doc["metadata"])
        for child in children:
            all_chunks.append({"text": child.text, "metadata": {**child.metadata, "parent_id": child.parent_id}})
    print(f"  ✓ {len(all_chunks)} chunks from {len(docs)} documents ({time.time()-t0:.1f}s)", flush=True)

    # Step 2: Enrichment (M5)
    t0 = time.time()
    print(f"\n[2/4] Enriching {len(all_chunks)} chunks (M5, 1 API call/chunk)...", flush=True)
    enriched = enrich_chunks(all_chunks)
    if enriched:
        all_chunks = [
            {"text": e.enriched_text, "metadata": {**e.auto_metadata, "original_text": e.original_text}}
            for e in enriched
        ]
        print(f"  ✓ Enriched {len(enriched)} chunks ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("  ⚠️  M5 not implemented — using raw chunks", flush=True)

    # Step 3: Index (M2)
    t0 = time.time()
    print(f"\n[3/4] Indexing {len(all_chunks)} chunks (BM25 + Dense)...", flush=True)
    search = HybridSearch()
    reuse_dense = not rebuild_dense and search.dense.collection_ready()
    search.index(all_chunks, index_dense=not reuse_dense)
    action = "Reused Qdrant vectors; BM25 rebuilt" if reuse_dense else "Indexed"
    print(f"  ✓ {action} ({time.time()-t0:.1f}s)", flush=True)

    # Step 4: Reranker (M3)
    t0 = time.time()
    print("\n[4/4] Loading reranker...", flush=True)
    reranker = CrossEncoderReranker()
    print(f"  ✓ Reranker ready ({time.time()-t0:.1f}s)", flush=True)

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    results = search.search(query)
    docs = [{"text": r.text, "score": r.score, "metadata": r.metadata} for r in results]
    reranked = reranker.rerank(query, docs, top_k=RERANK_TOP_K)
    selected = reranked if reranked else results[:3]
    contexts = [item.metadata.get("original_text", item.text) for item in selected]

    from config import OPENAI_ANSWERS_ENABLED, OPENAI_API_KEY
    if OPENAI_ANSWERS_ENABLED and OPENAI_API_KEY and contexts:
        try:
            from openai import OpenAI
            client = OpenAI()
            context_str = "\n\n".join(contexts)
            resp = client.chat.completions.create(model="gpt-4o-mini", messages=[
                {"role": "system", "content": "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'"},
                {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
            ])
            answer = resp.choices[0].message.content
        except Exception as e:
            print(f"  ⚠️  LLM generation failed: {e}", flush=True)
            answer = contexts[0]
    else:
        answer = _extractive_answer(query, contexts)
    return answer, contexts


def _extractive_answer(query: str, contexts: list[str], max_sentences: int = 4) -> str:
    """Create a concise offline answer from the most query-relevant sentences."""
    if not contexts:
        return "Không tìm thấy thông tin."
    query_tokens = set(re.findall(r"\w+", query.lower(), flags=re.UNICODE))
    candidates = []
    for context_index, context in enumerate(contexts):
        sentences = [
            sentence.strip(" #>-\t")
            for sentence in re.split(r"(?<=[.!?])\s+|\n+", context)
            if len(sentence.strip(" #>-\t")) > 15
        ]
        ranked = []
        for sentence_index, sentence in enumerate(sentences):
            sentence_tokens = set(re.findall(r"\w+", sentence.lower(), flags=re.UNICODE))
            overlap = len(query_tokens & sentence_tokens) / max(len(query_tokens), 1)
            number_bonus = 0.15 if re.search(r"\d", sentence) and re.search(r"bao nhiêu|mấy|khoảng", query.lower()) else 0.0
            ranked.append((overlap + number_bonus, sentence_index, sentence))
        candidates.extend((score, context_index, index, sentence) for score, index, sentence in sorted(ranked, reverse=True)[:2])

    selected = []
    seen = set()
    for score, context_index, sentence_index, sentence in sorted(candidates, reverse=True):
        normalized = sentence.lower()
        if normalized in seen or score <= 0:
            continue
        seen.add(normalized)
        selected.append((context_index, sentence_index, sentence))
        if len(selected) >= max_sentences:
            break
    if not selected:
        return contexts[0]
    selected.sort(key=lambda item: (item[0], item[1]))
    return " ".join(sentence for _, _, sentence in selected)


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    test_set = load_test_set()
    print(f"\n[Eval] Running {len(test_set)} queries...", flush=True)
    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        print(f"  [{i+1}/{len(test_set)}] {item['question'][:50]}...", flush=True)

    t0 = time.time()
    print(f"\n[Eval] Evaluating 4 metrics × {len(test_set)} questions...", flush=True)
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)
    mode = results.get("evaluation_mode", "unknown")
    print(f"  ✓ Evaluation done ({time.time()-t0:.1f}s, mode={mode})", flush=True)

    print("\n" + "=" * 60)
    print("PRODUCTION RAG SCORES")
    print("=" * 60)
    for m in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        s = results.get(m, 0)
        print(f"  {'✓' if s >= 0.75 else '✗'} {m}: {s:.4f}")

    failures = failure_analysis(results.get("per_question", []))
    os.makedirs("reports", exist_ok=True)
    save_report(results, failures, path="reports/ragas_report.json")
    return results


if __name__ == "__main__":
    start = time.time()
    search, reranker = build_pipeline()
    evaluate_pipeline(search, reranker)
    print(f"\nTotal: {time.time() - start:.1f}s")
