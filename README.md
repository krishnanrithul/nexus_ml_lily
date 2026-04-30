# NexusML Lily 🌸

> An agentic RAG pipeline that makes post-earnings stock prediction model outputs interrogable in plain English — built with LangGraph, Claude API, and a three-tier retrieval system backed by LanceDB.

---

## What It Does

Synopsys runs an ML model that predicts stock price movements following earnings announcements. The model produces predictions, SHAP feature attribution values, and input feature snapshots — but these outputs lived in CSVs accessible only to the data science team.

NexusML Lily sits on top of those outputs and lets anyone ask:

- *"Which quarter had the worst prediction error?"*
- *"What factors drove the model's view on 2024Q3?"*
- *"Were macro factors like VIX driving our misses, or was it analyst signals?"*
- *"Which specific earnings events caught the model most off guard and why?"*

And get cited, grounded answers like:

> *"2024Q2 had the highest prediction error with RMSE 13.85 and MAPE 2.2% [1]. The largest misses were driven by earnings sentiment (+1.55) and VIX (-0.83) [2]."*

---

## Architecture

### Pipeline (LangGraph)

```
Manager (Claude)
    ├── Wrangler    — LLM-generated cleaning across 3 input CSVs
    ├── Analyzer    — Computes metrics from pre-computed outputs + SHAP
    └── Chronicler  — Generates narrative, indexes 3 record types to LanceDB
```

The manager reads full pipeline state after each stage and routes to the next agent. On failure, it diagnoses the error and injects a targeted fix into the retry prompt — **diagnosis-informed self-correction**, not blind retries.

### Three-Tier RAG (LanceDB)

| Tier | Record Type | Answers |
|------|-------------|---------|
| 1 | `narrative` | Overall accuracy, top features, model summary |
| 2 | `segment_summary` | Per-quarter RMSE/MAPE, best/worst periods |
| 3 | `prediction` | Individual quarter-day misses with SHAP drivers embedded |

Intent routing classifies the question before retrieval — segment questions only search `segment_summary` records, SHAP questions only search `prediction` records. Dilution avoided.

### Key Design Decisions

- **Reasoning over state, not hardcoded logic** — the LLM reads full context and decides the next action
- **Three-tier RAG** — separating aggregate narratives, segment summaries, and row-level predictions makes qualitatively different query types answerable from the same store
- **Diagnosis-informed retry** — passing a structured diagnosis on failure enables targeted fixes rather than hope-based retries
- **SHAP embedded in chunk text** — feature drivers are part of the retrieval document, not a separate lookup, so the LLM can cite them inline

---

## Data Inputs

| File | Key Columns | Purpose |
|------|-------------|---------|
| `prediction_results.csv` | `quarter, reaction_days, predicted_price, actual_excess` | Core model outputs per earnings cycle |
| `shap_features.csv` | `quarter_day, feature, shap_value` | Pre-computed SHAP attribution per feature |
| `input_features.csv` | `stock_close, vix, cpi, buy_ratings, avg_price_target` | Raw input snapshot at earnings date |

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| Orchestration | LangGraph |
| LLM | Claude API (`claude-sonnet-4-6`) |
| Vector Store | LanceDB |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Data | pandas, numpy |
| Language | Python 3.11+ |

---

## Getting Started

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Set your Claude API key

```bash
export ANTHROPIC_API_KEY=your_key_here
```

### 3. Add your data

Drop your three CSVs into `data/raw/`:

```
data/raw/
├── prediction_results.csv
├── shap_features.csv
└── input_features.csv
```

### 4. Run the pipeline

```bash
python main.py
```

### 5. Query your results

```bash
python query_engine.py
```

Or run the full demo question set:

```bash
python demo.py
```

---

## Example Session

```
You: Which quarter had the worst prediction error?

[intent: segment | sources: 3]
Nexus: Based on the segment summaries, 2024Q2 had the highest prediction
error with an RMSE of 13.85 and MAPE of 2.2% [1]. This was followed by
2024Q4 with RMSE 12.79 [2]. 2024Q3 was the most accurate quarter with
RMSE 7.61 and MAPE of just 1.1% [3].

You: What caused the large errors in 2024Q2?

[intent: prediction | sources: 5]
Nexus: The largest misses in 2024Q2 were primarily driven by earnings
sentiment (+1.55) and guidance surprise (+1.23), suggesting the model
underestimated the impact of analyst signal shifts that quarter [1][2].
VIX also contributed negatively (-0.83), indicating elevated macro
uncertainty [3].
```

---

## Project Structure

```
nexus_ml_lily/
├── main.py                  # Entry point — builds and runs the LangGraph pipeline
├── query_engine.py          # Interactive CLI + RAG query interface
├── demo.py                  # Runs director-level demo questions, saves to file
├── src/
│   ├── state.py             # FactoryState + Claude client wrapper
│   ├── manager.py           # LLM orchestrator with deterministic guards
│   └── workers/
│       ├── wrangler.py      # LLM-generated CSV cleaning (3 files)
│       ├── analyzer.py      # Metrics computation + SHAP aggregation
│       └── chronicler.py    # Narrative generation + LanceDB indexing
├── src/tools/
│   └── vector_ops.py        # LanceDB read/write — unchanged from NexusML
├── data/
│   └── raw/                 # Drop your 3 CSVs here
└── db/                      # LanceDB vector store (auto-created, gitignored)
```

---

## Relation to NexusML

NexusML Lily is an adaptation of [NexusML](https://github.com/krishnanrithul/nexus-ml) — a general-purpose agentic ML pipeline originally built for housing price prediction. The core architecture (LangGraph state machine, three-tier RAG, diagnosis-informed retry) is preserved. What changed:

- `modeler` → `analyzer` — no training; reads pre-computed financial model outputs
- SHAP values read from CSV rather than computed — feature attribution is richer and embedded directly in retrieval chunk text
- Domain language updated throughout for financial/earnings context
- `wrangler` handles three CSVs instead of one, with a shared `quarter_day` join key

---

*Built by Rithul Krishnan — Senior Staff Data Scientist, Synopsys*