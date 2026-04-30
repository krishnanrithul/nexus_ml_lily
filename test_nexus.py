import os
from src.state import FactoryState
from src.manager import manager_node
from src.workers.wrangler import wrangler_node
from nexus_ml_lily.src.workers.analyzer import modeler_node
from src.workers.chronicler import chronicler_node
from src.tools.vector_ops import clear_table, table_stats, get_all_chunks

WORKER_MAP = {
    "wrangler":       wrangler_node,
    "wrangler_retry": wrangler_node,
    "modeler":        modeler_node,
    "chronicler":     chronicler_node,
}


def run_test():
    print("🚀 Starting NexusML Integration Test...")

    # Clean slate — wipe any chunks from previous runs so retrieval results
    # reflect only this run's data
    print("\n🗑️  Clearing LanceDB table from previous runs...")
    clear_table()

    state: FactoryState = {
        "raw_data_path":    "data/raw/house_prices.csv",
        "cleaned_data_path": None,
        "model_results":    {},
        "report_chunks":    [],
        "messages":         [],
        "next_step":        "",
        "errors":           [],
        "retry_count":      0,
        "manager_diagnosis": "none",
    }

    max_steps = 10
    step = 0

    while step < max_steps:
        step += 1
        print(f"\n{'='*50}")
        print(f"STEP {step}")
        print(f"{'='*50}")

        manager_decision = manager_node(state)
        state.update(manager_decision)
        next_step = state["next_step"]

        if next_step == "end":
            print("\n✅ Pipeline completed successfully.")
            break

        if next_step not in WORKER_MAP:
            print(f"❌ Unknown next_step: '{next_step}'. Terminating.")
            break

        worker_fn = WORKER_MAP[next_step]
        print(f"▶️  Running worker: {next_step}")
        worker_output = worker_fn(state)
        state.update(worker_output)

        if state.get("errors"):
            print(f"⚠️  Errors in state: {state['errors']}")

    else:
        print(f"\n❌ Hit max_steps ({max_steps}) — likely an infinite loop. Check manager routing.")

    # ─────────────────────────────────────────────
    # FINAL STATE SUMMARY
    # ─────────────────────────────────────────────
    print("\n" + "="*50)
    print("FINAL STATE SUMMARY")
    print("="*50)

    # 1. Wrangler check
    cleaned = state.get("cleaned_data_path")
    if cleaned and os.path.exists(cleaned):
        print(f"✅ [WRANGLER] Cleaned file at       : {cleaned}")
    elif cleaned:
        print(f"❌ [WRANGLER] Path set but missing  : {cleaned}")
    else:
        print(f"❌ [WRANGLER] No cleaned_data_path in state")

    # 2. Modeler check
    model_results = state.get("model_results", {})
    if model_results:
        print(f"✅ [MODELER]  Best model             : {model_results.get('best_model')}")
        print(f"             Target column          : {model_results.get('target_column')}")
        print(f"             LR  RMSE / R²          : {model_results.get('lr_rmse', 'n/a'):.2f} / {model_results.get('lr_r2', 'n/a'):.4f}")
        print(f"             RF  RMSE / R²          : {model_results.get('rf_rmse', 'n/a'):.2f} / {model_results.get('rf_r2', 'n/a'):.4f}")
    else:
        print(f"❌ [MODELER]  No model_results in state")

    # 3. Chronicler check
    chunks = state.get("report_chunks", [])
    if chunks:
        print(f"✅ [CHRONICLER] Chunks in state      : {len(chunks)}")
        print(f"               First chunk preview  : {chunks[0][:100]}...")
    else:
        print(f"❌ [CHRONICLER] No report_chunks in state")

    # 4. LanceDB check
    print()
    stats = table_stats()
    if stats.get("status") == "ok":
        print(f"✅ [LANCEDB]  Total chunks indexed   : {stats['total_chunks']}")
        print(f"             Models stored          : {stats['unique_models']}")
        print(f"             Targets stored         : {stats['unique_targets']}")
        print(f"             Timestamps             : {stats['timestamps']}")

        # Spot-check retrieval with a canned question
        print("\n🔍 Spot-check: querying 'which model performed best?'")
        from src.tools.vector_ops import query_chunks
        results = query_chunks("which model performed best?", top_k=3)
        if results:
            for i, r in enumerate(results):
                print(f"  [{i+1}] (dist={r.get('_distance', 'n/a'):.4f}) {r['text'][:120]}")
        else:
            print("  ❌ No results returned from query.")
    else:
        print(f"❌ [LANCEDB]  {stats}")

    # 5. Messages + errors
    print(f"\n📨 Messages : {state.get('messages')}")
    print(f"❗ Errors   : {state.get('errors')}")


if __name__ == "__main__":
    run_test()