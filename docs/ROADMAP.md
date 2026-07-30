# Roadmap

What CellForge is missing, roughly in the order that fixing it would help. This is a
working document — [open an issue](https://github.com/gersteinlab/CellForge/issues/new/choose)
to argue with it.

Legend: 🔴 not started · 🟡 in progress · 🟢 done

---

## Now — lowering the cost of the first run

The gap between "this looks interesting" and "I ran it" is where most people leave.

| | Item | Why |
|---|---|---|
| 🔴 | **Colab / notebook demo** | A hosted notebook that runs Task Analysis on a small dataset with one API key. No conda, no npm, no cluster. This is the single highest-leverage thing on this list. |
| 🔴 | **Pre-computed example outputs** | Commit one real analysis report, research plan, and `result.py` under `examples/outputs/` so people can see what CellForge produces before spending a token. |
| 🔴 | **`--dry-run`** | Walk the pipeline with a stub LLM, print what *would* be called and the estimated token cost. Catches configuration mistakes for free. |
| 🔴 | **Cost estimate before the expensive stage** | Print projected tokens and dollars before Method Design starts, and ask. |
| 🔴 | **Docker image** | `docker run cellforge` with the scientific stack baked in. The full pip install is several GB and a common first-run failure. |
| 🔴 | **PyPI release** | `pip install cellforge`. Requires deciding what the minimum runtime dependency set actually is. |

## Next — making the results harder to argue with

| | Item | Why |
|---|---|---|
| 🔴 | **More datasets** | Six is defensible, not conclusive. Replogle, Dixit, Sci-Plex extensions, and any primary-tissue perturbation atlas would each widen coverage. See [DATASETS.md](DATASETS.md#adding-a-dataset-to-the-benchmark). |
| 🔴 | **A leaderboard** | Published splits, a scoring script, and a table anyone can submit to. Turns a paper result into a living benchmark, which is what actually attracts sustained contribution. |
| 🔴 | **Committed baseline implementations** | The paper compares against 12 baselines. None of them are in this repo. Vendoring or scripting them makes every future comparison cheap. |
| 🔴 | **Deterministic replay** | Record every LLM call from a run and replay it offline. Gives exact reproducibility for debugging without claiming the *design* is deterministic. |
| 🔴 | **Held-out design evaluation** | Run CellForge on a dataset published after the model's training cutoff. The strongest available answer to "is it just recalling the literature?" |

## Later — capability

| | Item | Why |
|---|---|---|
| 🔴 | **Iterative refinement from results** | Right now the loop ends when the model is trained. Feeding metrics back into Method Design for a second design round is the obvious next stage, and closest to what a human actually does. |
| 🔴 | **More code-generation backends** | Claude Code, Aider, OpenHands as a first-class (non-legacy) option. The [registry](../cellforge/Code_Generation/registry.py) is small; this is a good first PR. |
| 🔴 | **More retrieval providers** | bioRxiv, OpenAlex, Europe PMC. All unclaimed. |
| 🔴 | **Beyond perturbation prediction** | Cell-type annotation, trajectory inference, spatial. The framework is general; the prompts and corpus are not. |
| 🔴 | **Multi-objective design** | Let the debate optimise accuracy *and* interpretability, or accuracy under a parameter budget. |
| 🔴 | **Prompt-injection defences for retrieved text** | Currently unmitigated and documented as such in [SECURITY.md](../SECURITY.md). |

## Engineering debt

| | Item | Why |
|---|---|---|
| 🔴 | **Prune `legacy/`** | `cellforge/legacy/rag_v1/` and `Code_Generation/legacy/` are dead weight in the import path. Move to a branch or a tag. |
| 🔴 | **Split `requirements.txt`** | One file currently pins the LLM stack, the scientific stack, Jupyter, and web scraping together. Nobody needs all of it. Extras: `[retrieval]`, `[scanpy]`, `[dev]`. |
| 🔴 | **Type coverage** | `mypy` is configured and unenforced. Start with `cellforge/retrieval/` and `Code_Generation/`. |
| 🔴 | **Integration test with a stub LLM** | The 43 unit tests never exercise a full pipeline run. A fake provider returning canned responses would cover the orchestration. |
| 🔴 | **Rename `figs/Bioforge_workflow.png`** | Leftover from an earlier project name. |
| 🟢 | **CI** | Test matrix on 3.9–3.12, lint, build, citation validation, secret scan. |
| 🟢 | **Contributor documentation** | [CONTRIBUTING](../CONTRIBUTING.md), [SECURITY](../SECURITY.md), issue and PR templates. |

---

## What we are not going to do

Saying no is part of a roadmap:

- **Ship pretrained weights.** CellForge produces models; it is not a model zoo. Shipping
  weights would make people use the frozen artifact instead of the process, which is the
  opposite of the point.
- **Guarantee reproducible architectures.** Determinism here would mean freezing the
  design, and the design being open-ended is the contribution. Deterministic *replay*
  for debugging is on the list; deterministic *design* is not.
- **Remove the human checkpoints.** Fully autonomous GPU submission is a small
  convenience and a large risk. The plan gate and the script gate stay.
- **A general-purpose research agent.** CellForge is for computational method design in
  single-cell omics. Scope creep would make it worse at that.

---

## Helping

Pick anything 🔴 and open an issue saying you are on it, so two people do not do the
same work. If you want to be told what to do, in descending order of value:

1. Add a dataset — [how](DATASETS.md#adding-a-dataset-to-the-benchmark)
2. Add a baseline
3. Build the Colab demo
4. Add a code-generation backend or a retrieval provider
5. [Report a bad plan](https://github.com/gersteinlab/CellForge/issues/new?template=bad_plan.yml)

[CONTRIBUTING.md](../CONTRIBUTING.md) has the details.
