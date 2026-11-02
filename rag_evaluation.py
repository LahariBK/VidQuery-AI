
import re
import time
import pandas as pd


def normalize_text(text):
    """Normalize text for simple word-overlap evaluation."""
    text = str(text).lower()
    return re.findall(r"\b[a-z0-9]+\b", text)


def calculate_overlap(expected_answer, actual_answer):
    """
    Calculate token overlap between expected and actual answers.
    This is a simple lexical metric, not a semantic correctness score.
    """
    expected_tokens = set(normalize_text(expected_answer))
    actual_tokens = set(normalize_text(actual_answer))

    if not expected_tokens:
        return 0.0

    return len(expected_tokens & actual_tokens) / len(expected_tokens)


def evaluate_questions(
    test_cases,
    embedder,
    db,
    rag,
    top_k=4
):
    """
    Evaluate RAG on test cases.

    Each test case must contain:
      - question
      - expected_answer

    Optional:
      - evidence_text: transcript evidence expected to be retrieved
    """
    results = []

    for case in test_cases:
        question = str(case.get("question", "")).strip()
        expected = str(case.get("expected_answer", "")).strip()
        evidence = str(case.get("evidence_text", "")).strip()

        if not question or not expected:
            continue

        start = time.perf_counter()

        query_embedding = embedder.embed_query(question)
        retrieved_chunks = db.search(query_embedding, top_k=top_k)
        answer = rag.answer(retrieved_chunks, question)

        elapsed = time.perf_counter() - start

        retrieved_text = " ".join(
            str(chunk.get("text", ""))
            for chunk in retrieved_chunks
        )

        answer_overlap = calculate_overlap(expected, answer)

        if evidence:
            evidence_tokens = set(normalize_text(evidence))
            retrieved_tokens = set(normalize_text(retrieved_text))
            retrieval_hit = (
                len(evidence_tokens & retrieved_tokens) /
                len(evidence_tokens)
                if evidence_tokens else 0.0
            )
            retrieval_hit = retrieval_hit >= 0.5
        else:
            retrieval_hit = None

        results.append({
            "question": question,
            "expected_answer": expected,
            "generated_answer": answer,
            "retrieved_chunks": len(retrieved_chunks),
            "retrieval_hit": retrieval_hit,
            "answer_token_recall": round(answer_overlap, 3),
            "response_time_seconds": round(elapsed, 2),
            "evidence_text": evidence
        })

    return pd.DataFrame(results)
