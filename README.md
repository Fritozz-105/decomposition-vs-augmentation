# Decomposition vs Augmentation: Isolating the Impact of Multi-Agent Architectures in Entity Resolution

By Zachary Zeng (University of Florida)

Code, prompts, released prediction caches, and analysis scripts for the paper
*Decomposition vs Augmentation: Isolating the Impact of Multi-Agent
Architectures in Entity Resolution*.

## Description

The paper compares three entity resolution approaches on the
[Abt-Buy](https://huggingface.co/datasets/matchbench/Abt-Buy) product catalog
benchmark, evaluated on the same fixed 1,916-pair labeled test split (206 true
matches):

1. **Magellan-style Random Forest** - a traditional ML baseline with 14
   hand-crafted similarity features (Levenshtein, Jaro-Winkler, Jaccard,
   Needleman-Wunsch, Smith-Waterman, TF-IDF cosine, price features),
   implemented with scikit-learn and jellyfish.
2. **Single-LLM** - one `gpt-oss-120b` model that receives each product pair
   and returns a match/no-match verdict, with no tools and no agent
   decomposition.
3. **Multi-agent LLM** - a `gpt-oss-120b` orchestrator that dispatches each
   pair to two specialized reviewer agents, then issues the final verdict:
   - *Syntactic agent* - MCP tools for sequence dissimilarity, word-token
     Jaccard, and BM25 over product names.
   - *Semantic agent* - MCP tools for price comparison and
     `all-MiniLM-L6-v2` embedding cosine similarity.

Neither LLM configuration receives retrieval or external knowledge; the
multi-agent tools compute deterministic signals from the pair itself. This
isolates what agent decomposition alone contributes.

## Reproducing the Paper's Numbers (no API calls, no cost)

The five prediction caches per LLM pipeline are released under
`results/{single_llm,multi_agent}/eval_run_{0..4}` (the illustrative "Run 1"
in the paper is `eval_run_0`). Every LLM metric in the paper re-derives from
these caches:

```bash
git clone git@github.com:Fritozz-105/decomposition-vs-augmentation.git
cd decomposition-vs-augmentation
uv sync --group dev
uv run python -m pytest                          # 272 tests
uv run python -m scripts.compute_paper_stats     # Table 1 LLM rows, agreement, kappa, inference
```

`compute_paper_stats` validates that every cache holds exactly the 1,916
labeled test keys before computing anything, then writes
`results/paper_stats.json` (canonical) and `results/paper_stats.md`
(human-readable). Reruns are byte-identical. It also verifies the runtime
transcription against `time.txt` by digest.

Three further scripts read the same caches (they need the dataset present
locally; run `uv run python -m src.main blocking` once first to download it):

```bash
uv run python scripts/analyze_final_results.py   # Table 2 error categories
uv run python scripts/analyze_supplementary.py   # threshold optima, per-agent counterfactuals, PR curves
uv run python scripts/analyze_arbitration.py     # agent agreement and orchestrator arbitration counts
```

Their committed outputs are in `results/`.

## Re-running the Pipelines (costs API calls)

Full re-execution takes roughly 27 hours of sequential API calls
(~42 min/run single-LLM, ~281 min/run multi-agent) and is **not** guaranteed
to reproduce the released caches byte-for-byte: the serving build of
`gpt-oss-120b` behind the API at the original run time was not recorded.

### Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` with your credentials:

```text
OPENAI_API_KEY=your-navigator-api-key
NAVIGATOR_API_BASE=https://api.ai.it.ufl.edu/v1/    # UF NaviGator endpoint (default)
MODEL_NAME=gpt-oss-120b                             # Model used in the paper
HF_TOKEN=your-huggingface-token                     # Optional - raises HuggingFace rate limits
```

> **Note:** The paper's runs used the [UF NaviGator](https://navigator.ai.ufl.edu/)
> endpoint, which requires a UF account. Any OpenAI-compatible endpoint works:
> update `NAVIGATOR_API_BASE` and `MODEL_NAME` accordingly.

### Commands

```bash
uv run python -m src.main blocking                  # TF-IDF blocking (downloads dataset on first run)
uv run python -m src.main magellan-tune             # GridSearchCV; saves data/magellan_best_params.json
uv run python -m src.main magellan                  # Magellan-style RF
uv run python -m src.main single-llm                # Single-LLM pipeline
uv run python -m src.main multi-agent               # Multi-agent pipeline
uv run python -m src.main evaluate-all --runs 5     # Full evaluation, all pipelines
```

Each pipeline accepts `--max-pairs N` for a cheap smoke run.

## Repository Structure

```
├── paper/                  # LaTeX source (main.tex, citations.bib) + figures
├── src/
│   ├── main.py             # CLI entry point
│   ├── blocking/           # TF-IDF cosine blocking
│   ├── data/               # Abt-Buy loader (HuggingFace)
│   ├── pipelines/
│   │   ├── magellan/       # RF baseline: features, pipeline, tuning
│   │   ├── single_llm/     # Single-LLM baseline: prompts, pipeline
│   │   └── multi_agent/    # Agents, MCP tools, orchestrator consensus
│   ├── evaluation/         # Shared metrics, runner, results output
│   └── utils/              # LLM client, lookups, parsing, paths
├── scripts/                # Post-hoc analysis (see above)
├── tests/                  # pytest suite (272 tests)
├── results/                # Released caches + canonical statistics + reports
└── time.txt                # Raw timing log (provenance for runtime numbers)
```

## Reproducibility Notes

- Every reported metric is computed on the dataset's predefined 1,916-pair
  `test` split, passed directly to each matcher; TF-IDF blocking characterizes
  a deployment configuration and is not on the evaluated path.
- The dataset snapshot is not revision-pinned on HuggingFace; the released
  caches and the key-label digest in `results/paper_stats.json` identify
  exactly what was scored.
- Magellan run-to-run variability comes from the Random Forest
  `random_state`; LLM variability from residual API non-determinism at
  temperature 0.

## License

MIT - see [LICENSE](LICENSE).
