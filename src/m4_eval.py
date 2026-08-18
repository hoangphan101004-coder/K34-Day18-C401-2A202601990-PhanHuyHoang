from __future__ import annotations

"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json, math, re
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import OPENAI_API_KEY, RAGAS_ENABLED, TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    lengths = {len(questions), len(answers), len(contexts), len(ground_truths)}
    if len(lengths) != 1:
        raise ValueError("questions, answers, contexts and ground_truths must have equal lengths")
    if not questions:
        return _aggregate([], "empty")

    if _has_openai_key():
        try:
            from datasets import Dataset
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
            )

            dataset = Dataset.from_dict({
                "question": questions,
                "answer": answers,
                "contexts": contexts,
                "ground_truth": ground_truths,
            })
            result = evaluate(
                dataset,
                metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
            )
            rows = result.to_pandas()
            per_question = [
                EvalResult(
                    question=str(row["question"]),
                    answer=str(row["answer"]),
                    contexts=list(row["contexts"]),
                    ground_truth=str(row["ground_truth"]),
                    faithfulness=_number(row.get("faithfulness", 0.0)),
                    answer_relevancy=_number(row.get("answer_relevancy", 0.0)),
                    context_precision=_number(row.get("context_precision", 0.0)),
                    context_recall=_number(row.get("context_recall", 0.0)),
                )
                for _, row in rows.iterrows()
            ]
            return _aggregate(per_question, "ragas")
        except Exception as exc:
            print(f"  Warning: RAGAS evaluation failed ({exc}); using local lexical metrics.")

    per_question = [
        _evaluate_locally(question, answer, item_contexts, ground_truth)
        for question, answer, item_contexts, ground_truth in zip(
            questions, answers, contexts, ground_truths
        )
    ]
    return _aggregate(per_question, "local_lexical_fallback")


def _has_openai_key() -> bool:
    return bool(
        RAGAS_ENABLED
        and OPENAI_API_KEY
        and OPENAI_API_KEY.startswith("sk-")
        and OPENAI_API_KEY not in {"sk-...", "sk-your-key-here"}
        and len(OPENAI_API_KEY) > 20
    )


def _number(value) -> float:
    try:
        number = float(value)
        return 0.0 if math.isnan(number) else number
    except (TypeError, ValueError):
        return 0.0


def _tokens(text: str) -> set[str]:
    stopwords = {"và", "là", "có", "được", "của", "cho", "trong", "một", "những", "các", "theo"}
    return {
        token for token in re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        if len(token) > 1 and token not in stopwords
    }


def _coverage(source: set[str], evidence: set[str]) -> float:
    return len(source & evidence) / max(len(source), 1)


def _evaluate_locally(question: str, answer: str, contexts: list[str], ground_truth: str) -> EvalResult:
    answer_tokens = _tokens(answer)
    question_tokens = _tokens(question)
    truth_tokens = _tokens(ground_truth)
    context_token_sets = [_tokens(context) for context in contexts]
    all_context_tokens = set().union(*context_token_sets) if context_token_sets else set()

    relevant_contexts = [
        _coverage(truth_tokens, context_tokens) for context_tokens in context_token_sets
    ]
    return EvalResult(
        question=question,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        faithfulness=_coverage(answer_tokens, all_context_tokens),
        answer_relevancy=max(
            _coverage(question_tokens, answer_tokens),
            _coverage(truth_tokens, answer_tokens),
        ),
        context_precision=(
            sum(score >= 0.15 for score in relevant_contexts) / len(relevant_contexts)
            if relevant_contexts else 0.0
        ),
        context_recall=_coverage(truth_tokens, all_context_tokens),
    )


def _aggregate(results: list[EvalResult], mode: str) -> dict:
    metrics = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
    aggregate = {
        metric: sum(getattr(item, metric) for item in results) / len(results) if results else 0.0
        for metric in metrics
    }
    return {**aggregate, "evaluation_mode": mode, "per_question": results}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    diagnostic_tree = {
        "faithfulness": ("Câu trả lời có nội dung không được context hỗ trợ", "Siết prompt và chỉ cho phép trích dẫn từ context"),
        "context_recall": ("Retriever bỏ sót thông tin cần thiết", "Điều chỉnh chunking, BM25 hoặc mở rộng top-k"),
        "context_precision": ("Context chứa nhiều đoạn không liên quan", "Tăng chất lượng reranking hoặc lọc metadata"),
        "answer_relevancy": ("Câu trả lời chưa bám sát câu hỏi", "Cải thiện prompt trả lời và xử lý loại câu hỏi"),
    }
    analyzed = []
    for result in eval_results:
        scores = {
            "faithfulness": result.faithfulness,
            "answer_relevancy": result.answer_relevancy,
            "context_precision": result.context_precision,
            "context_recall": result.context_recall,
        }
        worst_metric = min(scores, key=scores.get)
        diagnosis, suggested_fix = diagnostic_tree[worst_metric]
        analyzed.append({
            "question": result.question,
            "answer": result.answer,
            "ground_truth": result.ground_truth,
            "worst_metric": worst_metric,
            "score": round(float(scores[worst_metric]), 4),
            "average_score": round(sum(scores.values()) / len(scores), 4),
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix,
        })
    return sorted(analyzed, key=lambda item: item["average_score"])[:max(bottom_n, 0)]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
