<div align="center">

# 🧬 CellForge

### Agentic design of virtual cell models

**Give CellForge a single-cell dataset and a research question. It reads the literature, argues with itself until a group of expert agents converge on an architecture, writes the training code, and runs it.**

[![arXiv](https://img.shields.io/badge/arXiv-2508.02276-b31b1b.svg?style=flat-square)](https://arxiv.org/abs/2508.02276)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg?style=flat-square)](https://www.python.org/downloads/)
[![CI](https://img.shields.io/github/actions/workflow/status/gersteinlab/CellForge/ci.yml?branch=main&style=flat-square&label=tests)](https://github.com/gersteinlab/CellForge/actions/workflows/ci.yml)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](CONTRIBUTING.md)
[![Stars](https://img.shields.io/github/stars/gersteinlab/CellForge?style=flat-square&color=gold)](https://github.com/gersteinlab/CellForge/stargazers)

[**Quickstart**](#-quickstart-5-minutes) · [**What it designed**](#-what-cellforge-designed) · [**Benchmarks**](#-benchmarks) · [**Architecture**](#-architecture) · [**Docs**](docs/) · [**Paper**](https://arxiv.org/abs/2508.02276)

</div>

---

## The one-minute version

Building a virtual cell model is a months-long loop: read the perturbation-modeling literature, pick an architecture, adapt it to your assay, write the training code, debug the CUDA errors, tune, evaluate, repeat.

CellForge collapses that loop into a single command.

```bash
cellforge --dataset-path data/datasets/adamson.h5ad \
          --task "Predict single-cell gene expression after CRISPRi knockdown in K562."
```

What comes back is not a chat transcript. It is **a research plan with citations, a runnable `result.py`, and a trained model** — and on the six perturbation benchmarks in the paper, the models it designed are competitive with or better than hand-built state of the art.

| | |
|---|---|
| **PCC 0.9883** | Adamson CRISPRi — best of all methods compared |
| **MSE_DE 0.1736** | Norman combinatorial CRISPRa, on differentially expressed genes |
| **14 / 15** | task–criterion combinations where blinded LLM judges ranked CellForge first |
| **6 datasets, 3 modalities** | scRNA-seq, scATAC-seq, CITE-seq — genetic, chemical, and cytokine perturbations |
| **4–8 GPU-hours** | end to end, question → trained model, on a single GPU |

> [!NOTE]
> CellForge does not ship pretrained weights. It ships the *process that produces them*. Every architecture below was specified by the agents, not selected from a menu.

<div align="center">
<img src="figs/CellForge_project_overview.svg" alt="CellForge workflow: inputs, task analysis, method design, code generation, autorun execution, outputs" width="100%">
</div>

---

## 🚀 Quickstart (5 minutes)

**1. Install**

```bash
git clone https://github.com/gersteinlab/CellForge.git
cd CellForge
conda create -n cellforge python=3.11 -y && conda activate cellforge
pip install -e .
```

**2. Add one API key**

```bash
cp .env.example .env
# Open .env and set at least one of:
#   OPENAI_API_KEY / ANTHROPIC_API_KEY / DEEPSEEK_API_KEY / QWEN_API_KEY / LLAMA_API_KEY
```

**3. Check your workspace**

```bash
cellforge --init      # create data/{datasets,analyses,plans,codes,literature}
cellforge --doctor    # verify config, dataset dir, literature dir, LLM key, Python version
```

**4. Get a dataset**

```bash
python scripts/download_datasets.py adamson --out data/datasets/
```

**5. Run it**

```bash
cellforge \
  --dataset-path data/datasets/adamson.h5ad \
  --task "Predict post-perturbation gene expression in K562 cells after CRISPRi knockdown. \
Report MSE, PCC, R2 overall and restricted to differentially expressed genes."
```

You will find:

```
data/
├── analyses/<dataset>/    # structured task analysis + retrieved literature with provenance
├── plans/<dataset>/       # the research plan the expert agents converged on
└── codes/<dataset>/       # result.py — verified, runnable training code
```

Want to also *train* the model it wrote? Add the opt-in execution stage:

```bash
cellforge --phase autorun --dataset-path data/datasets/adamson.h5ad --executor local --workers 2
```

📖 **Full walkthrough:** [docs/QUICKSTART.md](docs/QUICKSTART.md) · **Running on a cluster:** [docs/QUICKSTART.md#running-on-slurm](docs/QUICKSTART.md#running-on-slurm)

---

## 🏛 Architecture

CellForge is three stages plus an opt-in fourth. Each maps to a directory in the package, and each writes a human-readable artifact you can inspect, edit, or approve before the next stage runs.

```
                    ┌──────────────────────────────────────────────┐
   dataset (.h5ad)  │                                              │
        +           │  ①  TASK ANALYSIS      cellforge/Task_Analysis/
   task description │     ─────────────                            │
        ───────────▶│     Dataset Analyst   ·  Problem Investigator │
                    │     Baseline Assessor ·  Refinement Agent     │
                    │     + literature retrieval (local · PubMed ·  │
                    │       Crossref · Semantic Scholar · Qdrant)   │
                    │                    ↓                         │
                    │             analysis report                  │
                    │                    ↓                         │
                    │  ②  METHOD DESIGN     cellforge/Method_Design/
                    │     ─────────────                            │
                    │     Data-modeling expert                     │
                    │     Single-cell biology expert     ⟳ debate   │
                    │     Deep-learning expert           ⟳ review   │
                    │     Training expert                ⟳ revise   │
                    │          ↕ central CRITIC (area chair)       │
                    │                    ↓                         │
                    │             research plan    ◀── 👤 checkpoint │
                    │                    ↓                         │
                    │  ③  CODE GENERATION   cellforge/Code_Generation/
                    │     ─────────────                            │
                    │     coding agent edits result.py in place    │
                    │     → deterministic verifier (file / syntax / │
                    │       CLI / contract) → bounded repair ×5     │
                    │                    ↓                         │
                    │             result.py        ◀── 👤 checkpoint │
                    │                    ↓                         │
                    │  ④  AUTORUN (opt-in)  cellforge/autorun/      │
                    │     task-wise split → local or Slurm sbatch   │
                    └──────────────────────────────────────────────┘
                                          ↓
                              trained model + metrics
```

**The part that makes it work: agents that disagree.**

Method Design is not a prompt chain. Four domain experts each draft a proposal, then peer-review every other proposal, while a central Critic plays conference area chair. Each agent carries a **coordination score** updated every round:

$$c_t^{(i)} = 0.3\,c_{t-1}^{(i)} \;+\; 0.4\,r_{\text{crit},t}^{(i)} \;+\; \frac{0.3}{k-1}\sum_{j \neq i} r_{\text{peer},t}^{(i,j)}$$

Debate runs until every agent clears $c \ge 0.8$ **and** the widest pairwise gap falls below $0.03$ — consensus, not agreement-by-exhaustion — or until 20 rounds elapse. Ablating the Critic, the peer-review loop, or the retrieval corpus each measurably degrades the final model.

**Two human checkpoints, by design.** Nothing touches a GPU unattended: you approve the research plan after Method Design, and the training script before submission.

📖 **Deep dive:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 👀 See the output before you spend a token

A complex run costs roughly 80k prompt / 400k completion tokens, plus 4–8 GPU-hours if you train what it writes. You should be able to look at the deliverables first.

**[`examples/outputs/`](examples/outputs/)** is a complete worked bundle for the Adamson CRISPRi task — the task analysis, the research plan, and the training script, laid out exactly as a run leaves them:

| Stage | Artifact | |
|---|---|---|
| Task Analysis | [`task_analysis_report.md`](examples/outputs/adamson_crispri/analyses/task_analysis_report.md) | dataset, problem, baselines, agent refinement round |
| Method Design | [`research_plan.md`](examples/outputs/adamson_crispri/plans/research_plan.md) | architecture + protocol — 👤 **your approval gate** |
| Code Generation | [`result.py`](examples/outputs/adamson_crispri/workspace/result.py) | runnable training script — 👤 **approval before any GPU job** |
| Verification | [`verification.json`](examples/outputs/adamson_crispri/workspace/verification.json) | the deterministic, non-LLM acceptance check |

```bash
cd examples/outputs/adamson_crispri/workspace
python result.py --help        # no third-party dependencies needed
python result.py --selftest
```

> [!IMPORTANT]
> **The two documents are reference artifacts, not transcripts of a live run.** They are written to match the schemas the code serialises, section for section. `result.py` is real, working, verified code, and its `metrics.json` is the genuine output of really running it — on **synthetic** data, so those numbers are a smoke test and nothing more.
>
> Every file's provenance is stated individually in [`PROVENANCE.md`](examples/outputs/adamson_crispri/PROVENANCE.md). Real benchmark numbers are in [docs/RESULTS.md](docs/RESULTS.md).
>
> Have you run the pipeline for real? [`scripts/export_example_run.py`](scripts/export_example_run.py) packages and scrubs a run into a bundle — a real one should replace this, and we would take that PR gladly.

---

## 🔬 What CellForge designed

Six datasets in, six architectures out. None were templates — each was specified by the agents, then implemented and trained end to end. Full model cards: [docs/MODELS.md](docs/MODELS.md).

| Model | Designed for | The idea the agents landed on |
|---|---|---|
| **CPA-X** | Adamson CRISPRi (scRNA-seq) | Compositional perturbation autoencoder with a disentangled perturbation embedding and adversarial covariate removal |
| **scGen-X** | Norman combinatorial CRISPRa | Latent-space arithmetic extended to *pairs* of simultaneously activated genes, with an explicit interaction term for non-additive effects |
| **ChemCellFlow** | Srivatsan drug response | Sinkhorn conditional optimal transport coupled to a 6-layer normalizing flow — dose-aware and chemically conditioned |
| **CPA-Traj** | Schiebinger cytokine time course | Trajectory-aware VAE that conditions on time as a continuous covariate rather than a class label |
| **totalGAT** | Papalexi CITE-seq | Graph attention over the gene network, cross-attention between RNA and protein, separate decoder heads per modality |
| **ChromDDPM** | Liscovitch-Brauer scATAC-seq | Denoising diffusion over the 200k-peak accessibility profile, conditioned on the perturbation |

The trajectory-aware encoder in **CPA-Traj** and the diffusion denoiser in **ChromDDPM** have no counterpart in the seed literature corpus. They came out of the debate.

---

## 📊 Benchmarks

Six datasets, seven readouts, five-fold cross-validation, three independent runs, perturbation-centric averaging. Baselines: CPA, scGen, CondOT, Biolord, scGPT, GEARS, STATE, ChemCPA, CellFlow, random forest, linear regression, and the unperturbed control.

| Benchmark | CellForge model | Headline result |
|---|---|---|
| Adamson CRISPRi | CPA-X | **PCC 0.9883** — best of all methods compared |
| Norman combinatorial CRISPRa | scGen-X | **MSE_DE 0.1736** |
| Schiebinger cytokine time course | CPA-Traj | DEG recall 0.535 |
| Papalexi CITE-seq (protein) | totalGAT | protein recall 0.420 |

**Against other autonomous systems.** Blinded LLM judges (Claude 3.7, DeepSeek-R1, OpenAI o1, Qwen-plus, Llama 3.1) scored CellForge against OpenAI Deep Research, Perplexity Deep Research, Gemini Deep Research, Biomni, and a single-LLM baseline across 15 task–criterion combinations. **CellForge ranked first in 14 of 15.** Inter-judge agreement was Pearson 0.88–0.93; human expert ratings tracked the judge panel at r = 0.87 — versus r = 0.53 for the system's own internal confidence, which is precisely why the Critic is external.

⚠️ **Where it is still weak.** 0.535 DEG recall on the cytokine time course and 0.420 protein recall on Papalexi are honest numbers, not typos. Sparse, low-count modalities remain hard, and we would rather you know that before you start. Full tables — all 7 metrics, all baselines, all six datasets: [docs/RESULTS.md](docs/RESULTS.md).

---

## 🗂 Datasets

| Dataset | Perturbation | Modality | Cells / features | Accession |
|---|---|---|---|---|
| Adamson 2016 | CRISPRi | scRNA-seq | 111k / 33k genes | [GSE90546](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE90546) |
| Norman 2019 | combinatorial CRISPRa | scRNA-seq | 84k / 17k genes | [GSE133344](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE133344) |
| Srivatsan 2020 | small molecules | scRNA-seq | 81k / 18k genes | [GSE139944](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE139944) |
| Schiebinger 2019 | cytokine time course | scRNA-seq | 65k / 17k genes | scPerturb |
| Papalexi 2021 | CRISPR | CITE-seq | 171k / 18k genes + 200 proteins | scPerturb |
| Liscovitch-Brauer 2021 | CRISPR | scATAC-seq | 58k / 200k peaks | scPerturb |

Preprocessed `.h5ad` files for all six are mirrored by [scPerturb](https://projects.sanderlab.org/scperturb/) at DOI [10.5281/zenodo.13350497](https://doi.org/10.5281/zenodo.13350497).

```bash
python scripts/download_datasets.py --list           # show everything available
python scripts/download_datasets.py norman papalexi  # fetch specific ones
```

📖 **Preprocessing, splits, and DEG definitions:** [docs/DATASETS.md](docs/DATASETS.md)

---

## 🧩 Bring your own everything

| You want to swap… | How |
|---|---|
| **LLM provider** | Set `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`, `QWEN_API_KEY`, or `LLAMA_API_KEY` — or point `CUSTOM_API_URL` at any OpenAI-compatible endpoint (vLLM, Ollama, TGI, OpenRouter) |
| **Coding agent** | `--codegen-backend codex`; the backend registry in [`cellforge/Code_Generation/registry.py`](cellforge/Code_Generation/registry.py) takes a new backend in about 40 lines |
| **Literature corpus** | Drop PDFs in `CELLFORGE_LITERATURE_DIR`; set `QDRANT_ENABLED=true` for vector search over them |
| **Retrieval providers** | PubMed, Crossref, and Semantic Scholar are pluggable in [`cellforge/retrieval/providers.py`](cellforge/retrieval/providers.py) |
| **Compute** | `--executor local`, or `--executor slurm` with `--partition / --gres / --mem / --slurm-time` |
| **Cost ceiling** | `METHOD_DESIGN_MAX_ROUNDS`, `METHOD_DESIGN_MAX_EXPERTS`, `METHOD_DESIGN_MAX_TOKENS_PER_CALL` |

---

## 💰 What a run actually costs

Per end-to-end run, from the paper:

| | Simple task | Complex task |
|---|---|---|
| Prompt tokens | ~40k | ~80k |
| Completion tokens | ~200k | ~400k |
| Generated model size | 10–30M parameters | 10–30M parameters |
| Wall clock (1 GPU) | ~4h | ~8h |

For a cheap smoke test: `MODEL_NAME=gpt-4o-mini` with `METHOD_DESIGN_MAX_ROUNDS=2` and `METHOD_DESIGN_MAX_EXPERTS=2`.

**When it fails, this is how** — measured across runs: computation execution error 41% · invalid type or unsupported operation 23% · error-recovery failure 16% · model misconfiguration 6% · data access 5% · other system-level 5% · **hallucinated structures 4%**. Outright invented architecture is the *rarest* failure; the verifier and repair loop catch most of it. See [docs/FAQ.md](docs/FAQ.md#failure-modes).

---

## 🔐 Safety notes

- **Nothing runs on a GPU without you.** The research plan and the training script are both human-approval checkpoints.
- **Generated code is verified before it is published.** Files, Python syntax, CLI surface, and the acceptance contract are checked deterministically; failures loop back to the same agent workspace for up to five bounded repair attempts.
- **The coding agent runs in an OS sandbox** (`workspace-write`), with isolated credentials: `CODEX_AUTH_MODE=local` strips provider API keys from the subprocess. Task Analysis and Method Design credentials are never silently reused for code generation.
- **Agent traces may contain model-generated commands and output.** They are written with owner-only permissions under `<output_dir>/.cellforge_workspaces/`. Review a workspace before publishing it.

---

## 📚 Documentation

| | |
|---|---|
| [**Quickstart**](docs/QUICKSTART.md) | Install, configure, first run, Slurm |
| [**Example outputs**](examples/outputs/) | What the pipeline actually hands you, stage by stage |
| [**Architecture**](docs/ARCHITECTURE.md) | The stages, agent roster, coordination score, ablations |
| [**Results**](docs/RESULTS.md) | Full benchmark tables, all metrics, all baselines, judge study |
| [**Model cards**](docs/MODELS.md) | The six generated architectures in detail |
| [**Datasets**](docs/DATASETS.md) | Sources, preprocessing, splits, DEG definitions |
| [**FAQ**](docs/FAQ.md) | Cost, failure modes, limitations, troubleshooting |
| [**Roadmap**](docs/ROADMAP.md) | Where this is going, and what to help with |
| [**Contributing**](CONTRIBUTING.md) | Dev setup, tests, PR conventions |

---

## 🤝 Contributing

Good first issues, in rough order of how much they help:

1. **Add a dataset.** A new perturbation dataset with a loader and a split definition is the single most valuable contribution — it directly widens the benchmark.
2. **Add a baseline.** More comparators make the evaluation harder to argue with.
3. **Add a code-generation backend.** The registry is small on purpose.
4. **Add a retrieval provider.** bioRxiv, OpenAlex, and Europe PMC are all unclaimed.
5. **Report a failure.** A run that produced a bad plan, with the plan attached, is genuinely useful data.

```bash
pip install -e ".[dev]"
python -m pytest tests -v
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) first. Be decent: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md).

---

## 📖 Citation

```bibtex
@article{tang2025cellforge,
  title   = {CellForge: Agentic Design of Virtual Cell Models},
  author  = {Tang, Xiangru and Yu, Zhuoyun and Chen, Jiapeng and Cui, Yan and
             Shao, Daniel and Wang, Weixu and Wu, Fang and Zhuang, Yuchen and
             Shi, Wenqi and Huang, Zhi and Cohan, Arman and Lin, Xihong and
             Theis, Fabian and Krishnaswamy, Smita and Gerstein, Mark},
  journal = {arXiv preprint arXiv:2508.02276},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.02276}
}
```

A [`CITATION.cff`](CITATION.cff) is included, so GitHub's *Cite this repository* button works too.

---

## 📜 License

MIT — see [LICENSE](LICENSE).

<div align="center">
<br>

**Built at the [Gerstein Lab](https://gersteinlab.org), Yale University.**

If CellForge saved you a month of architecture search, a ⭐ is a nice way to say so.

</div>
