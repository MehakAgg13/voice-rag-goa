# backend/guardrails.py

import re


# Topics that are relevant to our MSMARCO-based RAG system.
# We keep this broad because the dataset contains general
# information-seeking questions.
ALLOWED_INTENT_KEYWORDS = {
    "what", "who", "when", "where", "why", "how",
    "which", "can", "does", "is", "are",
    "explain", "define", "meaning", "difference",
    "information", "tell", "describe", "find"
}


# Basic unsafe-content patterns.
# This is only a first layer; the LLM will have another
# safety/grounding check later.
UNSAFE_PATTERNS = [
    r"\bhow to (make|build|create)\s+(a\s+)?bomb\b",
    r"\bhow to (make|build|create)\s+(a\s+)?weapon\b",
    r"\bhow to (make|build|create)\s+(a\s+)?explosive\b",
    r"\bkill someone\b",
    r"\bhow to poison\b",
    r"\bmake poison\b",
]


def check_safety(query: str) -> tuple[bool, str]:
    """
    Check whether the user query contains obviously unsafe content.

    Returns:
        (True, "") if safe
        (False, reason) if unsafe
    """

    query = query.strip().lower()

    for pattern in UNSAFE_PATTERNS:
        if re.search(pattern, query):
            return False, "I can't help with unsafe or harmful instructions."

    return True, ""


def check_query(query: str) -> dict:
    """
    Main guardrail function.

    Returns a structured result so the rest of the pipeline
    knows whether it should continue to retrieval/LLM.
    """

    if not query or not query.strip():
        return {
            "allowed": False,
            "reason": "empty_query",
            "message": "I couldn't understand the question. Please try again."
        }

    query = query.strip()

    # Safety check
    is_safe, safety_message = check_safety(query)

    if not is_safe:
        return {
            "allowed": False,
            "reason": "unsafe",
            "message": safety_message
        }

    return {
        "allowed": True,
        "reason": "safe",
        "message": ""
    }


def check_grounding(retrieved_chunks: list, min_score: float = 0.35) -> dict:
    """
    Check whether retrieval returned enough relevant context.

    The exact score depends on the embedding/retrieval method.
    We keep this function independent so Member 1 can connect
    their retriever later.
    """

    if not retrieved_chunks:
        return {
            "grounded": False,
            "message": "I don't know based on the provided context."
        }

    # If retrieval results contain scores, check them.
    scores = []

    for chunk in retrieved_chunks:
        if isinstance(chunk, dict) and "score" in chunk:
            try:
                scores.append(float(chunk["score"]))
            except (ValueError, TypeError):
                pass

    if scores and max(scores) < min_score:
        return {
            "grounded": False,
            "message": "I don't know based on the provided context."
        }

    return {
        "grounded": True,
        "message": ""
    }