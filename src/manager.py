from src.state import client, FactoryState

# Valid actions the LLM can return
_VALID_ACTIONS = {"wrangler", "wrangler_retry", "analyzer", "analyzer_retry", "chronicler", "end"}

# Hard cap on retries — prevents infinite self-correction loops
_MAX_RETRIES = 2


def _build_routing_prompt(state: FactoryState) -> str:
    """
    Summarises the current pipeline state for the LLM to reason over.
    Gives the manager everything it needs to make a decision.
    """
    completed = []
    if state.get("validated_data"):
        vd = state["validated_data"]
        completed.append(
            f"- Data validated: {vd.get('results_rows')} prediction rows, "
            f"{vd.get('shap_rows')} SHAP rows, "
            f"{vd.get('input_rows')} input snapshots"
        )
    if state.get("analysis_results"):
        ar = state["analysis_results"]
        completed.append(
            f"- Analysis complete. Target: {ar.get('target_column')}, "
            f"Overall RMSE: {ar.get('overall_rmse', 'unknown'):.4f}, "
            f"Quarters analysed: {ar.get('n_quarters')}"
        )
    if state.get("report_chunks"):
        completed.append(f"- Narrative indexed: {len(state['report_chunks'])} chunks in LanceDB")

    completed_str = "\n".join(completed) if completed else "- Nothing completed yet"

    # Summarise errors
    error_str = "None"
    diagnosis_instruction = ""
    if state.get("errors"):
        last_error = state["errors"][-1]
        retry_count = state.get("retry_count", 0)
        error_str = f"{last_error} (retry attempt {retry_count} of {_MAX_RETRIES})"
        diagnosis_instruction = f"""
IMPORTANT: There is an active error. Before deciding the action, diagnose it:
- What specifically caused: "{last_error}"
- What one targeted fix should the worker apply on retry?
Include your diagnosis in the 'reasoning' field.
"""

    return f"""
You are the orchestration manager of NexusML, an agentic ML pipeline.
Your job is to assess the current state and decide what happens next.

PIPELINE STAGES (in order):
1. wrangler   — validates and loads the three input CSVs (prediction results, SHAP features, input features)
2. analyzer   — computes performance metrics from pre-computed model outputs and SHAP values
3. chronicler — generates narrative from results, indexes to LanceDB
4. end        — all stages complete

CURRENT STATE:
Completed steps:
{completed_str}

Last error: {error_str}
Messages: {state.get('messages', [])}
{diagnosis_instruction}
AVAILABLE ACTIONS:
- wrangler        : validate and load all three CSVs (first time)
- wrangler_retry  : re-run wrangler with error context (only if wrangler failed)
- analyzer        : compute metrics from prediction results and SHAP data
- analyzer_retry  : re-run analyzer with error context (only if analyzer failed)
- chronicler      : generate narrative and index to LanceDB
- end             : pipeline is complete or unrecoverable

Respond in this exact format, nothing else:
ACTION: <action>
REASONING: <one sentence explaining why>
DIAGNOSIS: <if there is an error, one sentence on the specific fix needed. Otherwise write 'none'>
"""


def _parse_response(response: str) -> tuple[str, str, str]:
    """
    Parses ACTION, REASONING, and DIAGNOSIS from the LLM response.
    Falls back gracefully if the model doesn't follow the format exactly.
    """
    action = "end"
    reasoning = "Could not parse response."
    diagnosis = "none"

    for line in response.strip().splitlines():
        line = line.strip()
        if line.startswith("ACTION:"):
            action = line.replace("ACTION:", "").strip().lower()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        elif line.startswith("DIAGNOSIS:"):
            diagnosis = line.replace("DIAGNOSIS:", "").strip()

    # Guard against hallucinated actions
    if action not in _VALID_ACTIONS:
        print(f"Manager: ⚠️ LLM returned unknown action '{action}'. Defaulting to 'end'.")
        action = "end"

    return action, reasoning, diagnosis


def manager_node(state: FactoryState) -> dict:
    """
    LLM-powered orchestrator. Reads full pipeline state, reasons about
    what to do next, and returns the action + a diagnosis for retries.
    """
    print("\n--- [MANAGER] Assessing System State ---")

    # Hard stop — if we've hit max retries don't even ask the LLM
    retry_count = state.get("retry_count", 0)
    if retry_count >= _MAX_RETRIES and state.get("errors"):
        print(f"Manager: 🛑 Max retries ({_MAX_RETRIES}) reached. Terminating.")
        return {"next_step": "end", "retry_count": 0}

    # Deterministic guards — check if stages are complete before asking LLM
    # This prevents the LLM from looping on stages that have already run

    if not state.get("validated_data"):
        print("Manager: No validated data. Routing to [WRANGLER]")
        return {"next_step": "wrangler", "retry_count": 0}

    if not state.get("analysis_results"):
        print("Manager: Data validated but no analysis. Routing to [ANALYZER]")
        return {"next_step": "analyzer", "retry_count": 0}

    if not state.get("report_chunks"):
        print("Manager: Analysis done but no narrative. Routing to [CHRONICLER]")
        return {"next_step": "chronicler", "retry_count": 0}

    # All stages complete
    print("Manager: ✅ All stages complete. Routing to END.")
    return {
        "next_step": "end",
        "retry_count": 0,
        "errors": [],
        "messages": ["Manager: Pipeline complete."],
    }