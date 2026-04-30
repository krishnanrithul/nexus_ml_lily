import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from src.state import client, FactoryState


# ── LLM helpers (preserve agentic reasoning) ──────────────────────────────────

def _infer_target_column(columns: list) -> str:
    """Ask the LLM which column represents the ground truth to evaluate against."""
    prompt = f"""
You are a Data Scientist reviewing a financial ML predictions dataset.
Column names: {columns}

This dataset contains pre-computed model predictions for stock price movements.
Which single column is the TARGET variable — the ground truth the model was trying to predict?

Likely candidates: actual_excess (actual excess return %), actual_price (actual stock price).
Reply with ONLY the exact column name. No explanation. No punctuation.
"""
    response  = client.generate(model='mistral', prompt=prompt)
    candidate = response['response'].strip().strip('"').strip("'")
    if candidate in columns:
        return candidate
    print(f"Analyzer: ⚠️ LLM returned '{candidate}' — not valid. Falling back to 'actual_excess'.")
    return "actual_excess"


def _infer_segment_column(columns: list) -> str:
    """Ask the LLM which column is the best segment dimension for performance analysis."""
    prompt = f"""
You are a Data Scientist analysing financial model predictions broken down by time period.
Column names: {columns}

Which single column should be used to SEGMENT performance analysis?
The best segment column groups rows into meaningful comparison buckets (e.g. by quarter, by reaction day).

Reply with ONLY the exact column name. No explanation. No punctuation.
"""
    response  = client.generate(model='mistral', prompt=prompt)
    candidate = response['response'].strip().strip('"').strip("'")
    if candidate in columns:
        return candidate
    print(f"Analyzer: ⚠️ LLM returned '{candidate}' — falling back to 'quarter'.")
    return "quarter"


# ── Metrics ────────────────────────────────────────────────────────────────────

def _evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    mape = float(np.mean(np.abs((y_true - y_pred) / np.where(y_true == 0, 1, y_true))) * 100)
    return {"rmse": round(rmse, 6), "r2": round(r2, 4), "mape": round(mape, 4)}


def _segment_metrics(df: pd.DataFrame, y_true: np.ndarray,
                     y_pred: np.ndarray, seg_col: str) -> dict:
    """
    Compute RMSE + MAPE per segment value (quarter or reaction_day).
    Returns: {"2024Q3": {"rmse": 0.42, "mape": 12.4, "n": 10}, ...}
    """
    results = {}
    for seg_val in df[seg_col].unique():
        mask = (df[seg_col] == seg_val).values
        if mask.sum() < 2:
            continue
        m = _evaluate(y_true[mask], y_pred[mask])
        results[str(seg_val)] = {"rmse": m["rmse"], "mape": m["mape"], "n": int(mask.sum())}
    return results


# ── SHAP aggregation ───────────────────────────────────────────────────────────

def _top_shap_features(df_shap: pd.DataFrame, quarter_day: str, n: int = 3) -> list[dict]:
    """
    For a given quarter_day, return the top-n features by absolute SHAP value.
    Returns: [{"feature": "guidance_surprise_pct", "shap_value": 1.55}, ...]
    """
    subset = df_shap[df_shap["quarter_day"] == quarter_day].copy()
    if subset.empty:
        return []
    subset["abs_shap"] = subset["shap_value"].abs()
    top = subset.nlargest(n, "abs_shap")[["feature", "shap_value"]]
    return top.to_dict(orient="records")


def _aggregate_shap(df_shap: pd.DataFrame) -> dict:
    """
    Global feature importance: mean absolute SHAP per feature across all quarter_days.
    Returns top 10 sorted descending.
    """
    agg = (df_shap.groupby("feature")["shap_value"]
                  .apply(lambda x: x.abs().mean())
                  .sort_values(ascending=False)
                  .head(10))
    return {feat: round(float(val), 4) for feat, val in agg.items()}


# ── Row-level prediction records ───────────────────────────────────────────────

def _build_row_predictions(df: pd.DataFrame, y_true: np.ndarray,
                            y_pred: np.ndarray, target_col: str,
                            seg_col: str, df_shap: pd.DataFrame) -> list[dict]:
    """
    One dict per row — stored as 'prediction' records in LanceDB.
    Enriched with top-3 SHAP drivers so the RAG layer can answer
    'why did the model miss 2024Q3 day 8?' not just 'what was the error?'
    """
    records = []
    for i in range(len(y_true)):
        actual    = float(y_true[i])
        predicted = float(y_pred[i])
        error_pct = abs(actual - predicted) / (abs(actual) if actual != 0 else 1) * 100

        row = df.iloc[i]
        quarter_day = row.get("quarter_day", "unknown")
        top_shap    = _top_shap_features(df_shap, quarter_day)

        records.append({
            "actual":      round(actual, 6),
            "predicted":   round(predicted, 6),
            "error_pct":   round(error_pct, 4),
            "target":      target_col,
            "segment":     str(row.get(seg_col, "unknown")),
            "quarter_day": quarter_day,
            "reaction_day": int(row.get("reaction_days", -1)),
            "top_shap":    top_shap,   # passed to chronicler for richer chunk text
        })
    return records


# ── Main node ──────────────────────────────────────────────────────────────────

def analyzer_node(state: FactoryState):
    """
    Worker: Analyses pre-computed model predictions. Produces three levels
    of output for the RAG layer — identical tier structure to the original
    modeler, adapted for financial prediction data:

      1. Aggregate metrics   → overall RMSE/MAPE/R² across all quarters
      2. Segment metrics     → per-quarter (or per-reaction-day) breakdown
      3. Row-level records   → individual quarter_day predictions + SHAP drivers

    LLM is used to infer target column and segment dimension — preserving
    agentic reasoning without needing to train any models.
    """
    paths    = state["data_paths"]
    is_retry = state.get("next_step") == "analyzer_retry"

    print("Analyzer: 📊 Loading prediction results and SHAP features...")

    try:
        df         = pd.read_csv(paths["results"])
        df_shap    = pd.read_csv(paths["shap"])
        bool_cols  = df.select_dtypes(include="bool").columns
        df[bool_cols] = df[bool_cols].astype(int)

        # Derive quarter_day if wrangler didn't persist it
        if "quarter_day" not in df.columns:
            df["quarter_day"] = df["quarter"] + "_day" + df["reaction_days"].astype(int).astype(str)

        if is_retry:
            diagnosis = state.get("manager_diagnosis", "none")
            print(f"Analyzer: ⚠️ Retrying. Manager diagnosis: {diagnosis}")

        # ── LLM infers target + segment ────────────────────────
        columns    = df.columns.tolist()
        target_col = _infer_target_column(columns)
        seg_col    = _infer_segment_column(columns)
        print(f"Analyzer: 🎯 Target='{target_col}'  Segment='{seg_col}'")

        pred_col = "predicted_excess" if target_col == "actual_excess" else "predicted_price"
        if pred_col not in df.columns:
            raise ValueError(f"Expected predicted column '{pred_col}' not found. Columns: {columns}")

        y_true = df[target_col].values
        y_pred = df[pred_col].values

        # ── Aggregate metrics ──────────────────────────────────
        overall = _evaluate(y_true, y_pred)
        print(f"Analyzer: Overall — RMSE {overall['rmse']:.4f}  "
              f"R² {overall['r2']:.4f}  MAPE {overall['mape']:.2f}%")

        # ── Segment metrics ────────────────────────────────────
        seg_results = _segment_metrics(df, y_true, y_pred, seg_col)
        if seg_results:
            worst    = max(seg_results, key=lambda s: seg_results[s]["rmse"])
            best_seg = min(seg_results, key=lambda s: seg_results[s]["rmse"])
            print(f"Analyzer: 📍 Segment '{seg_col}': "
                  f"worst={worst} (RMSE {seg_results[worst]['rmse']:.4f}), "
                  f"best={best_seg} (RMSE {seg_results[best_seg]['rmse']:.4f})")

        # ── SHAP feature importance (pre-computed, read from CSV) ──
        feat_imp = _aggregate_shap(df_shap)
        print(f"Analyzer: 🔍 Top 3 SHAP features: {list(feat_imp.keys())[:3]}")

        # ── Row-level predictions + SHAP drivers ──────────────
        row_preds = _build_row_predictions(df, y_true, y_pred, target_col, seg_col, df_shap)
        print(f"Analyzer: Built {len(row_preds)} row-level prediction records")

        return {
            "analysis_results": {
                # Aggregate
                "overall_rmse":       overall["rmse"],
                "overall_r2":         overall["r2"],
                "overall_mape":       overall["mape"],
                "target_column":      target_col,
                "predicted_column":   pred_col,
                "n_quarters":         len(seg_results),
                "n_rows":             len(df),
                # Segment level
                "segment_column":     seg_col,
                "segment_results":    seg_results,
                # Feature importance (from SHAP CSV)
                "feature_importance": feat_imp,
                # Row level — passed to chronicler for LanceDB indexing
                "row_predictions":    row_preds,
            },
            "errors":   [],
            "messages": [f"Analyzer: Complete. "
                         f"Overall RMSE={overall['rmse']:.4f}, "
                         f"{len(seg_results)} segments, "
                         f"{len(row_preds)} prediction records."],
        }

    except Exception as e:
        error_msg = f"Analyzer Error: {str(e)}"
        print(f"❌ {error_msg}")
        return {"errors": [error_msg]}