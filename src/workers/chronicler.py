import re
from datetime import datetime
from src.state import client, FactoryState
from src.tools.vector_ops import index_chunks


def _generate_narrative(analysis_results: dict) -> str:
    """LLM writes a 5-7 sentence summary of prediction performance and SHAP drivers."""
    seg     = analysis_results.get("segment_results", {})
    seg_col = analysis_results.get("segment_column", "quarter")

    seg_summary = ""
    if seg:
        worst    = max(seg, key=lambda s: seg[s]["rmse"])
        best_seg = min(seg, key=lambda s: seg[s]["rmse"])
        seg_summary = (f"{seg_col.capitalize()} '{worst}' had the highest prediction error "
                       f"(RMSE {seg[worst]['rmse']:.4f}, MAPE {seg[worst]['mape']:.1f}%) "
                       f"while '{best_seg}' was the most accurate "
                       f"(RMSE {seg[best_seg]['rmse']:.4f}, MAPE {seg[best_seg]['mape']:.1f}%).")

    top_features = list(analysis_results.get("feature_importance", {}).keys())[:3]

    prompt = f"""
You are a Senior Quantitative Analyst writing a concise model performance report for internal use.

Results:
- Target variable: {analysis_results['target_column']} (ground truth excess return or price)
- Predicted variable: {analysis_results['predicted_column']}
- Total predictions analysed: {analysis_results['n_rows']} rows across {analysis_results['n_quarters']} quarters
- Overall RMSE: {analysis_results['overall_rmse']:.4f}
- Overall R²: {analysis_results['overall_r2']:.4f}
- Overall MAPE: {analysis_results['overall_mape']:.2f}%
- Top 3 features by mean absolute SHAP value: {top_features}
- {seg_summary}

Write 5-7 sentences covering: what was being predicted, overall accuracy, 
which quarters performed best and worst, what the top SHAP features reveal about 
the model's behaviour, and one practical recommendation for improving predictions.
Professional prose only. No bullet points. No markdown.
"""
    response = client.generate(model='mistral', prompt=prompt)
    return response['response'].strip()


def _split_sentences(text: str) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 20]


def _build_narrative_chunks(narrative: str, analysis_results: dict,
                             timestamp: str) -> list[dict]:
    """
    TYPE 1 — narrative sentences.
    Good for: 'how did the model do overall?', 'what drove predictions?',
              'which quarter was most accurate?'
    """
    sentences = _split_sentences(narrative)
    return [
        {
            "text":           s,
            "record_type":    "narrative",
            "target_column":  analysis_results["target_column"],
            "best_model":     "pre-computed",
            "segment":        "all",
            "error_pct":      -1.0,
            "timestamp":      timestamp,
        }
        for s in sentences
    ]


def _build_segment_chunks(analysis_results: dict, timestamp: str) -> list[dict]:
    """
    TYPE 2 — one record per quarter (or reaction_day) with metrics as natural language.
    Good for: 'which quarter had the worst predictions?',
              'what was the MAPE for 2024Q3?'
    """
    seg_results = analysis_results.get("segment_results", {})
    seg_col     = analysis_results.get("segment_column", "quarter")
    chunks      = []

    for seg_name, metrics in seg_results.items():
        text = (
            f"{seg_col.capitalize()} '{seg_name}': "
            f"RMSE {metrics['rmse']:.4f}, "
            f"MAPE {metrics['mape']:.1f}%, "
            f"{metrics['n']} predictions. "
            f"Target: {analysis_results['target_column']}."
        )
        chunks.append({
            "text":           text,
            "record_type":    "segment_summary",
            "target_column":  analysis_results["target_column"],
            "best_model":     "pre-computed",
            "segment":        seg_name,
            "error_pct":      float(metrics["mape"]),
            "timestamp":      timestamp,
        })

    return chunks


def _build_prediction_chunks(analysis_results: dict, timestamp: str) -> list[dict]:
    """
    TYPE 3 — one record per quarter_day prediction, enriched with SHAP drivers.
    Good for: 'why did the model miss 2024Q3 day 8?',
              'show high-error predictions driven by VIX'

    Only indexes rows where error > 5% — high-error rows are what analysts ask about.
    SHAP drivers are embedded directly in chunk text so the LLM can cite them.
    """
    row_preds = analysis_results.get("row_predictions", [])
    chunks    = []

    for row in row_preds:
        if row["error_pct"] < 5.0:
            continue

        # Build SHAP driver string — e.g. "guidance_surprise_pct (+1.55), vix (-0.83)"
        shap_str = ""
        if row.get("top_shap"):
            parts = [
                f"{d['feature']} ({'+' if d['shap_value'] >= 0 else ''}{d['shap_value']:.2f})"
                for d in row["top_shap"]
            ]
            shap_str = f" Top SHAP drivers: {', '.join(parts)}."

        text = (
            f"Quarter {row['segment']}, reaction day {row['reaction_day']} "
            f"(quarter_day={row['quarter_day']}): "
            f"actual {analysis_results['target_column']}={row['actual']:.4f}, "
            f"predicted={row['predicted']:.4f}, "
            f"error={row['error_pct']:.1f}%."
            f"{shap_str}"
        )
        chunks.append({
            "text":           text,
            "record_type":    "prediction",
            "target_column":  row["target"],
            "best_model":     "pre-computed",
            "segment":        row["segment"],
            "error_pct":      float(row["error_pct"]),
            "timestamp":      timestamp,
        })

    return chunks
def _build_feature_chunks(analysis_results: dict, timestamp: str) -> list[dict]:
    """
    Indexes ALL features with their SHAP values — not just top 3.
    Enables questions like 'is sentiment used?' or 'what is the VIX contribution?'
    """
    feat_imp = analysis_results.get("feature_importance", {})
    chunks   = []

    for feature, shap_val in feat_imp.items():
        text = (
            f"Feature '{feature}' has a mean absolute SHAP value of {shap_val:.4f}. "
            f"Target: {analysis_results['target_column']}."
        )
        chunks.append({
            "text":          text,
            "record_type":   "narrative",
            "target_column": analysis_results["target_column"],
            "best_model":    "pre-computed",
            "segment":       "all",
            "error_pct":     -1.0,
            "timestamp":     timestamp,
        })

    return chunks

def chronicler_node(state: FactoryState):
    """
    Worker: Generates narrative + indexes THREE record types to LanceDB.

    record_type = 'narrative'        → aggregate performance story (5-7 sentences)
    record_type = 'segment_summary'  → one per quarter with RMSE/MAPE
    record_type = 'prediction'       → one per high-error quarter_day with SHAP drivers

    Three-tier structure unchanged from NexusML housing version —
    only the domain language and SHAP enrichment are new.
    """
    analysis_results = state["analysis_results"]
    timestamp        = datetime.utcnow().isoformat()

    print("Chronicler: ✍️  Generating narrative...")

    try:
        # 1. Narrative
        narrative        = _generate_narrative(analysis_results)
        narrative_chunks = _build_narrative_chunks(narrative, analysis_results, timestamp)
        print(f"Chronicler: {len(narrative_chunks)} narrative chunks")

        # 2. Segment summaries (per quarter / reaction_day)
        segment_chunks = _build_segment_chunks(analysis_results, timestamp)
        print(f"Chronicler: {len(segment_chunks)} segment summary chunks")

        # 3. Row-level predictions (high-error only, with SHAP drivers)
        prediction_chunks = _build_prediction_chunks(analysis_results, timestamp)
        print(f"Chronicler: {len(prediction_chunks)} prediction chunks (error > 5%)")
        
        #4. Feature chunks
        feature_chunks   = _build_feature_chunks(analysis_results, timestamp)
        print(f"Chronicler: {len(feature_chunks)} feature chunks")
        
        all_chunks = narrative_chunks + segment_chunks + prediction_chunks + feature_chunks
        index_chunks(all_chunks, {
            "target_column": analysis_results["target_column"],
            "best_model":    "pre-computed",
            "timestamp":     timestamp,
        })

        report_chunks = [c["text"] for c in narrative_chunks]

        return {
            "report_chunks": report_chunks,
            "errors":        [],
            "messages":      [f"Chronicler: Indexed {len(all_chunks)} chunks "
                              f"({len(narrative_chunks)} narrative, "
                              f"{len(segment_chunks)} segments, "
                              f"{len(prediction_chunks)} predictions)."],
        }

    except Exception as e:
        error_msg = f"Chronicler Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"errors": [error_msg]}