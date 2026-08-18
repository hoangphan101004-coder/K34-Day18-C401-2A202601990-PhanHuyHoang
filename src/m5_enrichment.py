from __future__ import annotations

"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import json, os, re, sys
from dataclasses import dataclass, field
from functools import lru_cache

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, OPENAI_ENRICHMENT_ENABLED


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.
    """
    if _api_enabled():
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt."},
                    {"role": "user", "content": text},
                ],
                max_tokens=150,
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  Warning: OpenAI summarize failed ({exc}); using extractive fallback.")
    return _local_summary(text)


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).
    """
    if n_questions <= 0:
        return []
    if _api_enabled():
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": f"Tạo {n_questions} câu hỏi tiếng Việt mà đoạn văn có thể trả lời. Mỗi dòng một câu hỏi."},
                    {"role": "user", "content": text},
                ],
                max_tokens=200,
                temperature=0,
            )
            questions = response.choices[0].message.content.strip().splitlines()
            cleaned = [re.sub(r"^\s*\d+[.)-]?\s*", "", q).strip() for q in questions if q.strip()]
            return cleaned[:n_questions]
        except Exception as exc:
            print(f"  Warning: OpenAI HyQA failed ({exc}); using extractive fallback.")
    return _local_questions(text, n_questions)


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).
    """
    context = ""
    if _api_enabled():
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Viết đúng 1 câu ngắn nêu nguồn và chủ đề của đoạn văn."},
                    {"role": "user", "content": f"Tài liệu: {document_title}\n\nĐoạn văn:\n{text}"},
                ],
                max_tokens=80,
                temperature=0,
            )
            context = response.choices[0].message.content.strip()
        except Exception as exc:
            print(f"  Warning: OpenAI contextual failed ({exc}); using source fallback.")
    if not context:
        context = _local_context(document_title)
    return f"{context}\n\n{text}" if context else text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.
    """
    if _api_enabled():
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": 'Trả về JSON hợp lệ: {"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}.'},
                    {"role": "user", "content": text},
                ],
                response_format={"type": "json_object"},
                max_tokens=150,
                temperature=0,
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as exc:
            print(f"  Warning: OpenAI metadata failed ({exc}); using rule-based metadata.")
    return _local_metadata(text)


# ─── Combined Single-Call Mode ───────────────────────────


def _enrich_single_call(text: str, source: str) -> dict:
    """Single LLM call to get summary + questions + context + metadata.

    ⚠️ Cost optimization: 1 API call thay vì 4 calls riêng lẻ.
    """
    if _api_enabled():
        try:
            response = _client().chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": """Phân tích đoạn văn và trả về JSON hợp lệ:
{"summary":"tóm tắt 2-3 câu","questions":["câu hỏi 1","câu hỏi 2","câu hỏi 3"],"context":"một câu nêu nguồn và chủ đề","metadata":{"topic":"...","entities":[],"category":"policy|hr|it|finance","language":"vi|en"}}"""},
                    {"role": "user", "content": f"Tài liệu: {source}\n\nĐoạn văn:\n{text}"},
                ],
                response_format={"type": "json_object"},
                max_tokens=400,
                temperature=0,
            )
            return _parse_json(response.choices[0].message.content)
        except Exception as exc:
            print(f"  Warning: OpenAI combined enrichment failed ({exc}); using local enrichment.")
    return {
        "summary": _local_summary(text),
        "questions": _local_questions(text, 3),
        "context": _local_context(source),
        "metadata": _local_metadata(text),
    }


def _api_enabled() -> bool:
    return bool(
        OPENAI_ENRICHMENT_ENABLED
        and OPENAI_API_KEY
        and OPENAI_API_KEY.startswith("sk-")
        and OPENAI_API_KEY not in {"sk-...", "sk-your-key-here"}
        and len(OPENAI_API_KEY) > 20
    )


@lru_cache(maxsize=1)
def _client():
    from openai import OpenAI

    return OpenAI(timeout=30.0, max_retries=1)


def _parse_json(content: str) -> dict:
    cleaned = content.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
    value = json.loads(cleaned)
    return value if isinstance(value, dict) else {}


def _local_summary(text: str) -> str:
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
    return " ".join(sentences[:2]) if sentences else text.strip()


def _local_questions(text: str, n_questions: int) -> list[str]:
    sentences = [s.strip(" #-\t") for s in re.split(r"[.!?\n]+", text) if len(s.strip()) > 10]
    return [f"Thông tin nào được quy định về {sentence[:100].rstrip()}?" for sentence in sentences[:n_questions]]


def _local_context(source: str) -> str:
    return f"Nội dung sau được trích từ tài liệu {source}." if source else "Nội dung sau thuộc kho chính sách nội bộ."


def _local_metadata(text: str) -> dict:
    lowered = text.lower()
    category = "policy"
    for candidate, keywords in {
        "it": ("mật khẩu", "vpn", "malware", "cntt", "dữ liệu"),
        "finance": ("lương", "chi phí", "tạm ứng", "thanh toán", "triệu", "vnđ"),
        "hr": ("nhân viên", "nghỉ phép", "thử việc", "bảo hiểm", "đào tạo"),
    }.items():
        if any(keyword in lowered for keyword in keywords):
            category = candidate
            break
    header = next((line.lstrip("# ").strip() for line in text.splitlines() if line.startswith("#")), "")
    topic = header or " ".join(text.strip().split()[:8]) or "general"
    return {"topic": topic[:120], "entities": [], "category": category, "language": "vi"}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks. (Đã implement sẵn — dùng functions ở trên)

    Có 2 chế độ:
    - methods cụ thể (["summary"], ["contextual"]...): gọi từng function riêng (tốt cho học/debug)
    - methods=["combined"] hoặc None: 1 API call duy nhất cho tất cả (tốt cho production)

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: Default None → combined mode (1 call/chunk).
                 Options: "summary", "hyqa", "contextual", "metadata", "combined"
    """
    if methods is None:
        methods = ["combined"]

    use_combined = "combined" in methods

    enriched = []
    for i, chunk in enumerate(chunks):
        text = chunk["text"]
        source = chunk.get("metadata", {}).get("source", "")

        if use_combined:
            result = _enrich_single_call(text, source)
            summary = result.get("summary", "")
            questions = result.get("questions", [])
            context_line = result.get("context", "")
            enrichment_parts = [context_line]
            if summary:
                enrichment_parts.append(f"Tóm tắt: {summary}")
            if questions:
                enrichment_parts.append("Câu hỏi liên quan: " + " ".join(questions))
            enrichment_parts.append(text)
            enriched_text = "\n\n".join(part for part in enrichment_parts if part)
            auto_meta = result.get("metadata", {})
        else:
            summary = summarize_chunk(text) if "summary" in methods else ""
            questions = generate_hypothesis_questions(text) if "hyqa" in methods else []
            enriched_text = contextual_prepend(text, source) if "contextual" in methods else text
            auto_meta = extract_metadata(text) if "metadata" in methods else {}

        enriched.append(EnrichedChunk(
            original_text=text,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata={**chunk.get("metadata", {}), **auto_meta},
            method="+".join(methods),
        ))

        if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
            print(f"  Enriched {i + 1}/{len(chunks)} chunks...", flush=True)

    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
