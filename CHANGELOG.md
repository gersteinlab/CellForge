# Changelog

All notable changes to CellForge are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Rewritten `README.md`: benchmark results, architecture diagram, model summaries,
  dataset table, cost and failure-mode figures, and safety notes.
- `docs/` — [Quickstart](docs/QUICKSTART.md), [Architecture](docs/ARCHITECTURE.md),
  [Results](docs/RESULTS.md), [Model cards](docs/MODELS.md),
  [Datasets](docs/DATASETS.md), [FAQ](docs/FAQ.md), [Roadmap](docs/ROADMAP.md).
- `scripts/download_datasets.py` — fetch the six benchmark datasets from scPerturb
  with checksum verification.
- `examples/` — ready-to-run task descriptions for each benchmark.
- GitHub Actions CI: test matrix on Python 3.9–3.12, lint, package build,
  `CITATION.cff` validation, and a secret scan.
- `requirements-ci.txt` — the minimal dependency set needed to run `pytest tests`,
  so CI does not install the full scientific stack.
- Issue templates (bug report, bad research plan, feature request), a pull-request
  template, and a Dependabot configuration.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`, and this changelog.
- `pyproject.toml` carrying the build backend and black / isort / ruff / mypy /
  pytest configuration.

### Changed
- `CITATION.cff` now lists the full author roster and points at
  [arXiv:2508.02276](https://arxiv.org/abs/2508.02276) as the preferred citation, so
  GitHub's *Cite this repository* button produces something usable.

## [0.1.0] — 2026-07-19

### Added
- Three-stage pipeline: Task Analysis → Method Design → Code Generation, plus an
  opt-in Autorun execution stage.
- Multi-agent Method Design with domain experts, peer review, and a central Critic
  scored by the coordination metric.
- Unified literature retrieval across a local PDF corpus, PubMed, Crossref, and
  Semantic Scholar, with deduplication by DOI / PMID / S2 ID and JSONL provenance.
- Codex code-generation backend with deterministic verification and a bounded
  five-attempt repair loop.
- Local and Slurm executors for task-wise experiment runs.
- `cellforge --init` and `cellforge --doctor` workspace management.

[Unreleased]: https://github.com/gersteinlab/CellForge/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/gersteinlab/CellForge/releases/tag/v0.1.0
