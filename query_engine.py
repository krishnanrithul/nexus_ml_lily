from src.state import client
from src.tools.vector_ops import (
    query_chunks, query_segments, query_predictions, table_stats
)


# ─────────────────────────────────────────────────────────────
# INTENT ROUTING
# ─────────────────────────────────────────────────────────────

_SEGMENT_KEYWORDS = {
    "quarter", "q1", "q2", "q3", "q4", "period", "earnings",
    "worst quarter", "best quarter", "which quarter", "which period",
    "worst", "best", "highest error", "lowest error", "which group",
    "reaction day", "day 1", "day 2", "day 3", "day 4", "day 5",
    "day 6", "day 7", "day 8", "day 9", "day 10",
}

_PREDICTION_KEYWORDS = {
    "row", "rows", "individual", "specific", "prediction", "predictions",
    "above", "below", "over", "under", "high error", "low error",
    "missed", "off by", "wrong", "worst case", "show me",
    "shap", "driver", "drivers", "what caused", "why did", "why was",
    "excess", "excess return", "quarter_day",
}


def _classify_intent(question: str) -> str:
    """
    Classify question into one of three retrieval intents.

    Returns:
        'segment'    → search segment_summary records (per-quarter metrics)
        'prediction' → search prediction records (quarter_day rows + SHAP drivers)
        'narrative'  → search narrative records (default — overall performance)
    """
    q = question.lower()

    segment_score    = sum(1 for kw in _SEGMENT_KEYWORDS    if kw in q)
    prediction_score = sum(1 for kw in _PREDICTION_KEYWORDS if kw in q)

    if segment_score == 0 and prediction_score == 0:
        return "narrative"
    if segment_score >= prediction_score:
        return "segment"
    return "prediction"


# ─────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────

def _retrieve(question: str, intent: str, top_k: int) -> list[dict]:
    """
    Route to the right retrieval function based on intent.
    Falls back to unfiltered search if the typed search returns nothing.
    """
    if intent == "segment":
        results = query_segments(question, top_k=top_k)
    elif intent == "prediction":
        results = query_predictions(question, top_k=top_k)
    else:
        results = query_chunks(question, top_k=top_k, record_type="narrative")

    if not results:
        print(f"[QueryEngine] No {intent} records found — falling back to full search.")
        results = query_chunks(question, top_k=top_k)

    return results


# ─────────────────────────────────────────────────────────────
# PROMPT BUILDING
# ─────────────────────────────────────────────────────────────

def _build_prompt(question: str, results: list[dict]) -> str:
    """
    Build RAG prompt with cited context.
    Each chunk is labelled with its record_type and segment so
    the LLM can reference the source in its answer.
    """
    context_lines = []
    for i, r in enumerate(results):
        rtype   = r.get("record_type", "unknown")
        segment = r.get("segment", "")
        dist    = r.get("_distance", None)

        label = f"[{i+1}] ({rtype}"
        if segment and segment not in ("all", "unknown"):
            label += f", quarter={segment}"
        if dist is not None:
            label += f", relevance={1 - dist:.2f}"
        label += ")"

        context_lines.append(f"{label} {r['text']}")

    context = "\n".join(context_lines)

    return f"""
You are a Quantitative Analyst assistant. Answer the question using ONLY the context below.
Each context item is labelled with its source type (narrative, segment_summary, or prediction)
and the quarter it relates to.
Cite the source label (e.g. [1], [2]) when you use it in your answer.
If the context doesn't contain enough information, say "I don't have data on that."
Never make up numbers, quarter names, SHAP values, or feature names not present in the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER (cite sources inline):
"""


# ─────────────────────────────────────────────────────────────
# PUBLIC API
# ─────────────────────────────────────────────────────────────

def ask(question: str, top_k: int = 5) -> tuple[str, str, list[dict]]:
    """
    Core RAG function. Returns answer, detected intent, and source chunks.

    Flow:
        1. Classify intent → which record type to search
        2. Retrieve top_k chunks from the right LanceDB subset
        3. Build cited prompt
        4. Generate answer with Mistral

    Args:
        question: Natural language question.
        top_k:    Chunks to retrieve. 5 for narrative/segment, 10 for predictions.

    Returns:
        (answer, intent, results)
        - answer:  string from Mistral
        - intent:  'narrative' | 'segment' | 'prediction'
        - results: list of retrieved chunk dicts (for display/debugging)
    """
    intent  = _classify_intent(question)
    k       = 10 if intent == "prediction" else top_k
    results = _retrieve(question, intent, top_k=k)

    if not results:
        return (
            "No results found. Run the pipeline first: python main.py",
            intent,
            []
        )

    prompt   = _build_prompt(question, results)
    response = client.generate(model="mistral", prompt=prompt)
    answer   = response["response"].strip()

    return answer, intent, results


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def run_chat():
    """
    Interactive CLI. Shows intent routing and sources for every answer
    so you can verify retrieval is working during development.
    """
    print("\n" + "="*58)
    print("  NexusML — Financial Prediction Query Engine")
    print("  Ask questions about your model's prediction results.")
    print("  Type 'help' for example questions. 'quit' to exit.")
    print("="*58)

    stats = table_stats()
    if stats.get("status") == "ok":
        by_type = stats.get("by_record_type", {})
        print(f"\n📚 Knowledge base ready — {stats['total_chunks']} total chunks")
        print(f"   narrative       : {by_type.get('narrative', 0)}")
        print(f"   segment_summary : {by_type.get('segment_summary', 0)}")
        print(f"   prediction      : {by_type.get('prediction', 0)}")
        print(f"   Targets : {stats['unique_targets']}")
        if stats.get("segments"):
            print(f"   Quarters: {stats['segments']}")
    else:
        print("\n⚠️  Knowledge base empty. Run first: python main.py")
        return

    example_questions = [
        "How accurate were the predictions overall?",
        "Which quarter had the worst prediction error?",
        "What were the most important SHAP features?",
        "Show me high-error predictions and what drove them.",
        "Why did the model miss 2024Q3?",
        "What was the MAPE for 2023Q2?",
        "Which reaction day had the highest error?",
        "What caused the large errors in 2024Q3 day 8?",
    ]

    print()

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in {"quit", "exit", "q"}:
            print("Goodbye.")
            break

        if question.lower() == "help":
            print("\nExample questions:")
            for q in example_questions:
                print(f"  → {q}")
            print()
            continue

        answer, intent, results = ask(question)

        print(f"\n[intent: {intent} | sources: {len(results)}]")
        print(f"Nexus: {answer}")

        if results:
            print("\n  Sources used:")
            for i, r in enumerate(results[:3]):
                dist     = r.get("_distance", None)
                dist_str = f" dist={dist:.3f}" if dist is not None else ""
                print(f"  [{i+1}] {r.get('record_type','?')}"
                      f"{dist_str} — {r['text'][:80]}...")
        print()


if __name__ == "__main__":
    run_chat()