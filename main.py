from langgraph.graph import StateGraph, END

from src.state import FactoryState
from src.manager import manager_node
from src.workers.wrangler import wrangler_node
from src.workers.analyzer import analyzer_node
from src.workers.chronicler import chronicler_node
from src.tools.vector_ops import clear_table, table_stats


# ─────────────────────────────────────────────
# ROUTER
# Called after every manager_node execution.
# Reads next_step from state and returns the
# name of the edge LangGraph should follow.
# ─────────────────────────────────────────────

def route(state: FactoryState) -> str:
    return state["next_step"]


# ─────────────────────────────────────────────
# GRAPH DEFINITION
# ─────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(FactoryState)

    # Register nodes
    graph.add_node("manager",    manager_node)
    graph.add_node("wrangler",   wrangler_node)
    graph.add_node("analyzer",   analyzer_node)
    graph.add_node("chronicler", chronicler_node)

    # Entry point — manager always runs first
    graph.set_entry_point("manager")

    # Manager decides what runs next via the router
    graph.add_conditional_edges(
        "manager",
        route,
        {
            "wrangler":        "wrangler",
            "wrangler_retry":  "wrangler",
            "analyzer":        "analyzer",
            "analyzer_retry":  "analyzer",
            "chronicler":      "chronicler",
            "end":             END,
        }
    )

    # Every worker reports back to the manager after completing
    graph.add_edge("wrangler",   "manager")
    graph.add_edge("analyzer",   "manager")
    graph.add_edge("chronicler", "manager")

    return graph.compile()


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────

def run(data_paths: dict, fresh: bool = True):
    """
    Run the full NexusML financial prediction pipeline.

    Args:
        data_paths: Dict with paths to the three input CSVs:
                    {
                        "results": "path/to/prediction_results.csv",
                        "shap":    "path/to/shap_features.csv",
                        "inputs":  "path/to/input_features.csv",
                    }
        fresh:      If True, clears LanceDB before running so results
                    reflect only this run. Set False to accumulate
                    results across multiple datasets.
    """
    if fresh:
        print("🗑️  Clearing previous LanceDB state...")
        clear_table()

    initial_state: FactoryState = {
        "data_paths":       data_paths,
        "validated_data":   None,
        "analysis_results": {},
        "report_chunks":    [],
        "messages":         [],
        "next_step":        "",
        "errors":           [],
        "retry_count":      0,
        "manager_diagnosis": "",
    }

    print(f"🚀 Starting NexusML pipeline")
    print(f"   results : {data_paths['results']}")
    print(f"   shap    : {data_paths['shap']}")
    print(f"   inputs  : {data_paths['inputs']}\n")

    graph = build_graph()
    final_state = graph.invoke(initial_state)

    # ── Summary ──────────────────────────────
    print("\n" + "="*50)
    print("PIPELINE COMPLETE")
    print("="*50)

    ar = final_state.get("analysis_results", {})
    if ar:
        print(f"✅ Target        : {ar.get('target_column')}")
        print(f"   Predicted     : {ar.get('predicted_column')}")
        print(f"   Overall RMSE  : {ar.get('overall_rmse'):.4f}")
        print(f"   Overall R²    : {ar.get('overall_r2'):.4f}")
        print(f"   Overall MAPE  : {ar.get('overall_mape'):.2f}%")
        print(f"   Quarters      : {ar.get('n_quarters')}")
        print(f"   Rows analysed : {ar.get('n_rows')}")
        top_feats = list(ar.get("feature_importance", {}).keys())[:3]
        print(f"   Top SHAP feats: {top_feats}")

    stats = table_stats()
    if stats.get("status") == "ok":
        by_type = stats.get("by_record_type", {})
        print(f"✅ LanceDB        : {stats['total_chunks']} chunks indexed")
        print(f"   narrative      : {by_type.get('narrative', 0)}")
        print(f"   segment_summary: {by_type.get('segment_summary', 0)}")
        print(f"   prediction     : {by_type.get('prediction', 0)}")

    if final_state.get("errors"):
        print(f"⚠️  Errors        : {final_state['errors']}")

    print("\n▶️  Run `python query_engine.py` to query your results.")

    return final_state


if __name__ == "__main__":
    run({
        "results": "data/raw/prediction_results.csv",
        "shap":    "data/raw/shap_features.csv",
        "inputs":  "data/raw/input_features.csv",
    })