# Tokenation_Hack

Inspect eval for LLM-driven experimental optimisation, using a Bayesian-optimisation tool and a legible reasoning graph.

- **Environment** (`bo_eval/env.py`): `upo_abts` uses `data/upo_abts.csv`. Each experiment snaps to the nearest measured condition and returns a draw from N(mean rate, SD).
- **Tools** (`bo_eval/tools/`): `run_experiment`, `bayes_opt_suggest` (GP + EI), `add_reasoning`, `close_branch` (Hintikka-style: the node and its descendants are closed and can't be extended), `view_graph`, `submit`.
- **Reasoning graph** (`bo_eval/state.py`): nodes are experiments (inputs and output), and edges carry reasoning. It is exported to `logs/graphs/*.md` (Mermaid) and `*.json`, and shown in the score explanation.
- **Scorer** (`bo_eval/scorer.py`): `found_optimal`, `n_experiments`, `regret`.
- **Solvers** (`bo_eval/solvers.py`): `react`, `react_no_bo`, `react_no_graph`, plus the no-LLM baselines `bo` and `random`. Add more to `SOLVERS`.

```bash
pip install -e .
inspect eval bo_eval/task.py --model openai/gpt-4o -T solver=react -T budget=30 --epochs 5
inspect eval bo_eval/task.py -T solver=bo --model mockllm/model
```
