
import time
import re
import pandas as pd

from models.llm import LLM
from rag_evaluation import calculate_overlap


MODELS_TO_COMPARE = [
    "phi3-smallctx:latest",
    "qwen2.5:0.5b",
]

FALLBACK_TEXT = "I couldn't find that information in the video."


def is_abstention(answer):
    """Detect common abstention phrases, including wording variations."""
    text = str(answer).lower()

    patterns = [
        r"i couldn't find",
        r"i could not find",
        r"not (?:available|present|mentioned|provided) in the (?:video|context|transcript)",
        r"(?:the )?(?:video|context|transcript) does not (?:contain|provide|mention)",
        r"there is not enough information",
        r"insufficient information",
        r"cannot determine from the (?:video|context|transcript)",
        r"can't determine from the (?:video|context|transcript)",
    ]

    return any(re.search(pattern, text) for pattern in patterns)


def calculate_binary_metrics(actual_labels, predicted_labels):
    """Positive class: answerable question."""
    tp = sum(a == 1 and p == 1 for a, p in zip(actual_labels, predicted_labels))
    fp = sum(a == 0 and p == 1 for a, p in zip(actual_labels, predicted_labels))
    fn = sum(a == 1 and p == 0 for a, p in zip(actual_labels, predicted_labels))
    tn = sum(a == 0 and p == 0 for a, p in zip(actual_labels, predicted_labels))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "true_negative": tn,
    }


def compare_models(test_cases, embedder, db, models=None, top_k=4):
    """
    Compare models using identical retrieved chunks for each question.

    answerable: 1 if the video supports answering the question; otherwise 0.

    grounding_label is intentionally left for manual review:
    Supported / Unsupported / Abstained / Mixed or Unclear.
    """
    models = models or MODELS_TO_COMPARE
    prepared_cases = []

    for case in test_cases:
        question = str(case.get("question", "")).strip()
        expected = str(case.get("expected_answer", "")).strip()
        evidence = str(case.get("evidence_text", "")).strip()

        try:
            label = int(case["answerable"])
        except (KeyError, ValueError, TypeError):
            continue

        if not question or not expected or label not in (0, 1):
            continue

        query_embedding = embedder.embed_query(question)
        chunks = db.search(query_embedding, top_k=top_k)
        context = "\n\n".join(
            str(chunk.get("text", "")) for chunk in chunks
        )

        prepared_cases.append({
            "question": question,
            "expected_answer": expected,
            "answerable": label,
            "evidence_text": evidence,
            "retrieved_chunks": len(chunks),
            "retrieved_context": context,
        })

    rows = []
    summaries = []
    llm = LLM()

    for model_name in models:
        llm.model = model_name
        predicted_labels = []

        for case in prepared_cases:
            start = time.perf_counter()
            error = ""

            try:
                answer = llm.generate(
                    case["retrieved_context"],
                    case["question"]
                )
            except Exception as exc:
                answer = ""
                error = str(exc)

            elapsed = time.perf_counter() - start
            abstained = is_abstention(answer)

            # A failed request is not a valid answerability prediction.
            predicted_answerable = (
                None if error else (0 if abstained else 1)
            )

            if predicted_answerable is not None:
                predicted_labels.append(predicted_answerable)

            # A provisional flag only. This is not a grounding verdict.
            if error:
                provisional_status = "Error"
            elif abstained:
                provisional_status = "Review abstention"
            else:
                provisional_status = "Needs manual grounding review"

            rows.append({
                "model": model_name,
                "question": case["question"],
                "answerable": case["answerable"],
                "expected_answer": case["expected_answer"],
                "generated_answer": answer,
                "retrieved_chunks": case["retrieved_chunks"],
                "retrieved_context": case["retrieved_context"],
                "evidence_text": case["evidence_text"],
                "abstained": abstained,
                "predicted_answerable": predicted_answerable,
                "grounding_label": "",
                "provisional_status": provisional_status,
                "answer_token_recall": calculate_overlap(
                    case["expected_answer"], answer
                ),
                "response_time_seconds": round(elapsed, 2),
                "error": error,
            })

        successful_rows = [
            row for row in rows
            if row["model"] == model_name and not row["error"]
        ]

        # Calculate answerability metrics only on successful requests.
        successful_cases = [
            case for case in prepared_cases
            if any(
                row["model"] == model_name
                and row["question"] == case["question"]
                and not row["error"]
                for row in rows
            )
        ]

        actual_labels = [case["answerable"] for case in successful_cases]
        predicted_successful = [
            row["predicted_answerable"] for row in successful_rows
        ]

        metrics = calculate_binary_metrics(
            actual_labels, predicted_successful
        )

        metrics["model"] = model_name
        metrics["mean_answer_token_recall"] = round(
            sum(row["answer_token_recall"] for row in successful_rows)
            / len(successful_rows), 3
        ) if successful_rows else 0.0

        metrics["average_response_time_seconds"] = round(
            sum(row["response_time_seconds"] for row in successful_rows)
            / len(successful_rows), 2
        ) if successful_rows else 0.0

        metrics["successful_requests"] = len(successful_rows)
        metrics["failed_requests"] = sum(
            row["model"] == model_name and bool(row["error"])
            for row in rows
        )
        summaries.append(metrics)

    return pd.DataFrame(rows), pd.DataFrame(summaries)
