# Contributing to CellForge

Thanks for being here. CellForge is a research system, so contributions land in two
quite different buckets, and they get reviewed differently:

- **Engineering** — bugs, backends, CLI, packaging, tests. Reviewed like normal software.
- **Science** — datasets, baselines, metrics, agent prompts, retrieval. Reviewed for
  whether the *result* is defensible, not just whether the code runs.

If you are unsure which bucket you are in, open an issue first and we will tell you.

---

## Development setup

```bash
git clone https://github.com/gersteinlab/CellForge.git
cd CellForge

conda create -n cellforge-dev python=3.11 -y
conda activate cellforge-dev

pip install -e ".[dev]"
```

The full install pulls the entire scientific stack (torch, scanpy, scvi-tools,
transformers). If you only want to run the unit suite, the lightweight path is:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-ci.txt
```

That is exactly what CI installs.

### Run the tests

```bash
python -m pytest tests -v
```

All 43 tests should pass in a couple of seconds. They are deliberately fast: no
network, no LLM calls, no GPU. If a test of yours needs any of those, mark it:

```python
@pytest.mark.network   # or @pytest.mark.llm, @pytest.mark.slow
def test_pubmed_provider_returns_records():
    ...
```

and run the default suite with `-m "not network and not llm and not slow"`.

### Formatting

```bash
black . && isort . && ruff check .
```

Config lives in [`pyproject.toml`](pyproject.toml). Line length is 100. The
`legacy/` directories are excluded on purpose — do not reformat them.

---

## The highest-value contributions

Ranked by how much they actually move the project:

### 1. Add a dataset

This is the best thing you can do. Every dataset you add widens the benchmark and
makes the evaluation harder to dismiss. A complete dataset contribution has:

- an entry in `scripts/download_datasets.py` with a stable URL and a checksum;
- a row in [`docs/DATASETS.md`](docs/DATASETS.md) — assay, perturbation type, cell
  and feature counts, accession;
- a documented split (which perturbations are held out, and why);
- at least one baseline number so the table is not empty.

### 2. Add a baseline

More comparators make the results credible. Please report the *tuned* baseline, not
a default-hyperparameter strawman — and say how you tuned it. A baseline PR that
beats CellForge on some dataset is welcome and will be merged; that is how a
benchmark stays honest.

### 3. Add a code-generation backend

The registry in [`cellforge/Code_Generation/registry.py`](cellforge/Code_Generation/registry.py)
is intentionally small. Implement the interface in
[`base.py`](cellforge/Code_Generation/base.py), register it, add a `--codegen-backend`
choice in `main.py`, and add a test alongside `tests/test_codegen_orchestrator.py`.

New backends must respect the existing safety properties: credentials isolated from
the general LLM keys, an OS-level sandbox, deterministic verification before any
generated file is published, and a bounded repair loop.

### 4. Add a retrieval provider

[`cellforge/retrieval/providers.py`](cellforge/retrieval/providers.py) currently
covers PubMed, Crossref, and Semantic Scholar. bioRxiv, OpenAlex, and Europe PMC are
unclaimed. Providers must deduplicate by DOI / PMID / S2 ID and emit a JSONL
provenance record per query — evidence without provenance is not evidence.

### 5. Report a bad plan

Use the [bad research plan](.github/ISSUE_TEMPLATE/bad_plan.yml) template. A run that
completed successfully but produced wrong science is more informative than a crash,
and we have nowhere else to get that data.

### 6. Contribute a real example bundle

[`examples/outputs/`](examples/outputs/) currently holds a *reference* bundle: the
documents are hand-written to match the schemas the code emits, because whoever
assembled it had no provider key and no downloaded dataset. A bundle from a genuine run
is strictly better, and replacing it is one of the most useful things you can do for
new users — it is the page people read while deciding whether to try this at all.

```bash
python scripts/export_example_run.py --dataset <name> --out examples/outputs/<name>
```

The script collects the artifacts, strips API keys, absolute paths, usernames and
hostnames, and leaves a `PROVENANCE.md` stub. Two rules:

- **Complete the provenance stub.** State what produced each file and what data the
  metrics came from. An unedited stub will be rejected.
- **Re-read every file yourself.** Scrubbing is best-effort, not a guarantee. Agent
  traces can echo secrets in shapes no regex anticipates.

If your metrics came from a subsample or from synthetic data, say so plainly rather than
letting them read as benchmark results.

---

## Changes to agent behaviour

Prompts, agent roles, the Critic, the coordination score, and the retrieval corpus all
determine what science comes out the other end. A PR that touches any of them needs
evidence, not just a green test run:

- the research plan produced **before** and **after**, on the same task and dataset; or
- a metric comparison on at least one benchmark, with the seed and split stated.

"It seems better" is not reviewable. We are not asking for a full ablation — one
concrete before/after is enough to have a conversation about.

---

## Pull requests

- Branch from `main`. One logical change per PR.
- Write a commit subject someone can read in a changelog: `feat: add Europe PMC
  retrieval provider`, not `update files`.
- Fill in the PR template, especially the "how it was verified" section.
- Add a `CHANGELOG.md` entry under **Unreleased** for anything user-visible.
- CI must be green. Lint jobs are advisory; the test job is not.

### Never commit

- API keys, tokens, or a filled-in `.env`
- Dataset files, model weights, or anything else large — link to it instead
- Agent workspaces from `.cellforge_workspaces/`; they contain model-generated
  command output and are written owner-only for a reason
- Absolute paths from your machine or your cluster

---

## Reporting security issues

Do not open a public issue. See [SECURITY.md](SECURITY.md).

## Code of conduct

By participating you agree to the [Code of Conduct](CODE_OF_CONDUCT.md).

## Licence

Contributions are accepted under the [MIT Licence](LICENSE).
