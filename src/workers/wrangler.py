import pandas as pd
import os
import re
from src.state import client, FactoryState

_RESULTS_COLS = {"quarter", "reaction_days", "current_price", "predicted_price",
                 "actual_price", "predicted_excess", "actual_excess"}
_SHAP_COLS    = {"quarter_day", "feature", "shap_value"}
_INPUTS_COLS  = {"stock_date", "stock_close", "vix", "cpi", "unemployment_rate",
                 "buy_ratings", "hold_ratings", "sell_ratings", "avg_price_target", "num_analysts"}

_FEATURE_NAME_MAP = {
    "cortex sentiment": "earnings sentiment",
    "cortex_sentiment": "earnings_sentiment",
}

def _clean_shap(df: pd.DataFrame) -> pd.DataFrame:
    df["shap_value"] = pd.to_numeric(df["shap_value"], errors="coerce")
    df["shap_value"] = df["shap_value"].fillna(0.0)
    df["feature"]    = df["feature"].replace(_FEATURE_NAME_MAP)
    return df

def _extract_code(raw_response: str) -> str:
    match = re.search(r"```(?:python)?\s*\n(.*?)```", raw_response, re.DOTALL)
    if match:
        return match.group(1).strip()
    lines = raw_response.strip().splitlines()
    code_lines = []
    in_code = False
    for line in lines:
        if re.match(r"^(import |from |def |class |#|if |for |while |try:|except|return|    |\w+ ?=)", line):
            in_code = True
        if in_code:
            code_lines.append(line)
    return "\n".join(code_lines).strip() if code_lines else raw_response.strip()


def _summarise(df: pd.DataFrame) -> dict:
    return {
        "shape":    df.shape,
        "dtypes":   df.dtypes.astype(str).to_dict(),
        "nulls":    df.isna().sum().to_dict(),
        "cat_cols": df.select_dtypes(include="object").columns.tolist(),
    }


def _build_prompt(s_results: dict, s_shap: dict, s_inputs: dict,
                  last_error: str = None, diagnosis: str = None) -> str:

    error_block = f"""
Your previous attempt failed with this error:
{last_error}

The orchestration manager diagnosed the issue as:
{diagnosis or "No specific diagnosis provided — fix the error above."}

Apply this specific fix. Return the full corrected function.
""" if last_error else ""

    return f"""
You are a Senior Data Engineer writing a Python data cleaning function for three financial ML output CSVs.
{error_block}

DATASET SUMMARIES:

prediction_results — {s_results['shape'][0]} rows
  dtypes : {s_results['dtypes']}
  nulls  : {s_results['nulls']}
  cat_cols: {s_results['cat_cols']}

shap_features — {s_shap['shape'][0]} rows
  dtypes : {s_shap['dtypes']}
  nulls  : {s_shap['nulls']}
  cat_cols: {s_shap['cat_cols']}

input_features — {s_inputs['shape'][0]} rows
  dtypes : {s_inputs['dtypes']}
  nulls  : {s_inputs['nulls']}
  cat_cols: {s_inputs['cat_cols']}

Write a function called `clean_data(df_results, df_shap, df_inputs)` that does exactly these steps in order:

STEP 1 — Fill numerical NaNs in each dataframe. Use EXACTLY this pattern for all three:
    for df in [df_results, df_shap, df_inputs]:
        num_cols = df.select_dtypes(include='number').columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

STEP 2 — Fill categorical NaNs in each dataframe. Use EXACTLY this pattern:
    for df in [df_results, df_shap, df_inputs]:
        for col in df.select_dtypes(include='object').columns:
            df[col] = df[col].fillna(df[col].mode()[0])

STEP 3 — Derive the quarter_day join key on df_results. Use EXACTLY this line:
    df_results['quarter_day'] = df_results['quarter'] + '_day' + df_results['reaction_days'].astype(int).astype(str)

STEP 4 — Coerce shap_value to numeric. Use EXACTLY this line:
    df_shap['shap_value'] = pd.to_numeric(df_shap['shap_value'], errors='coerce').fillna(0.0)

STEP 5 — Return all three cleaned dataframes as a tuple:
    return df_results, df_shap, df_inputs

BANNED — do not use any of these:
- df.replace() for filling NaNs
- sklearn, LabelEncoder, scipy, or any library other than pandas and numpy
- inplace=True on fillna applied to the whole dataframe at once
- Passing an Index object as a dict key anywhere

Return ONLY the function definition. No imports. No explanation. No markdown fences. No example calls.
"""


def wrangler_node(state: FactoryState):
    """
    Worker: LLM-generated cleaning across all three input CSVs, with self-correction.

    The LLM writes a clean_data(df_results, df_shap, df_inputs) function which is
    exec'd against the live dataframes. On failure, the manager's diagnosis is
    injected into the retry prompt so the LLM knows exactly what to fix.
    """
    paths    = state["data_paths"]
    is_retry = state.get("next_step") == "wrangler_retry"

    df_results = pd.read_csv(paths["results"])
    df_shap    = pd.read_csv(paths["shap"])
    df_inputs  = pd.read_csv(paths["inputs"])

    if is_retry:
        last_error = state["errors"][-1]
        diagnosis  = state.get("manager_diagnosis", "none")
        print(f"Wrangler: ⚠️ Self-correcting. Last error: {last_error}")
        print(f"Wrangler: 🔧 Manager diagnosis: {diagnosis}")
        prompt = _build_prompt(
            _summarise(df_results), _summarise(df_shap), _summarise(df_inputs),
            last_error=last_error, diagnosis=diagnosis
        )
    else:
        print("Wrangler: 🔍 Initial attempt — analysing all three CSVs...")
        prompt = _build_prompt(
            _summarise(df_results), _summarise(df_shap), _summarise(df_inputs)
        )

    response = client.generate(model='mistral', prompt=prompt)
    code     = _extract_code(response['response'])
    full_code = f"{code}\n\ndf_results, df_shap, df_inputs = clean_data(df_results, df_shap, df_inputs)"

    print(f"Wrangler: Generated code preview:\n{full_code[:400]}...")

    try:
        local_ns = {"pd": pd, "df_results": df_results, "df_shap": df_shap, "df_inputs": df_inputs}
        exec(full_code, local_ns)

        df_results = local_ns["df_results"]
        df_shap    = local_ns["df_shap"]
        df_inputs  = local_ns["df_inputs"]

        # Verify join key was derived correctly
        if "quarter_day" not in df_results.columns:
            raise ValueError("quarter_day join key missing from df_results after cleaning")

        overlap = set(df_results["quarter_day"]) & set(df_shap["quarter_day"])
        if not overlap:
            raise ValueError("No matching quarter_day keys between results and SHAP after cleaning")

        validated_data = {
            "results_rows":   len(df_results),
            "shap_rows":      len(df_shap),
            "input_rows":     len(df_inputs),
            "quarters":       sorted(df_results["quarter"].unique().tolist()),
            "n_quarter_days": len(df_results["quarter_day"].unique()),
            "join_overlap":   len(overlap),
        }

        print(f"Wrangler: ✅ results={validated_data['results_rows']} rows, "
              f"shap={validated_data['shap_rows']} rows, "
              f"inputs={validated_data['input_rows']} rows, "
              f"join overlap={validated_data['join_overlap']} quarter_day keys")

        return {
            "validated_data": validated_data,
            "errors":         [],
            "messages":       [f"Wrangler: {'Self-corrected' if is_retry else 'Cleaned'} successfully — "
                               f"{len(validated_data['quarters'])} quarters loaded."],
        }

    except Exception as e:
        error_msg = f"Wrangler Error: {str(e)}\n--- Generated Code ---\n{full_code}"
        print(f"❌ Wrangler failed: {error_msg}")
        return {"errors": [error_msg]}