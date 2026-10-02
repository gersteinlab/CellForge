# Security Policy

## Reporting a vulnerability

Please **do not open a public issue.** Use
[GitHub's private advisory form](https://github.com/gersteinlab/CellForge/security/advisories/new),
or email the maintainers listed on the [Gerstein Lab site](https://gersteinlab.org).

We aim to acknowledge reports within 5 working days.

## Supported versions

CellForge is pre-1.0 research software. Only `main` receives fixes.

## The threat model you should have in mind

CellForge runs an LLM-driven coding agent that writes and executes code on your
machine or cluster. That is the point of the tool, and it is also the main risk. The
design assumes the agent may produce wrong or hostile output and constrains it
accordingly:

| Control | What it does |
|---|---|
| **Human checkpoints** | The research plan and the generated training script are both approval gates. Nothing reaches a GPU unattended. |
| **Deterministic verification** | Generated code is checked for file presence, Python syntax, CLI surface, and acceptance-contract compliance before it is published as `result.py`. |
| **Bounded repair** | Verification failures loop back to the agent at most five times (`CODEGEN_MAX_REPAIR_ROUNDS`), then the run stops rather than retrying forever. |
| **OS sandbox** | The Codex backend runs under its `workspace-write` sandbox, so writes are confined to the task workspace. |
| **Credential isolation** | With the default `CODEX_AUTH_MODE=local`, provider API keys are stripped from the coding-agent subprocess. Task Analysis and Method Design credentials are never silently reused for code generation. Set `CODEX_AUTH_MODE=api` only if you deliberately want the fallback. |
| **Owner-only traces** | Agent workspaces under `<output_dir>/.cellforge_workspaces/` are written with restrictive permissions. |

### What is *not* protected

- **Prompt injection through retrieved literature.** CellForge ingests PDFs and search
  results into agent context. A hostile document could influence a research plan. This
  is a real limitation and one reason the plan is a human checkpoint.
- **The code CellForge writes is not audited for safety, only for validity.** Read
  `result.py` before running it, especially on shared infrastructure.
- **Agent traces may contain secrets.** Raw event streams, stderr, and final agent
  messages can echo environment contents. Review a workspace before publishing,
  attaching it to an issue, or committing it.
- **Your LLM provider sees your task descriptions and dataset metadata.** If your data
  is sensitive, use a self-hosted OpenAI-compatible endpoint via `CUSTOM_API_URL`.

## Good hygiene

- Keep `.env` out of git. It is already in `.gitignore` — verify before your first commit.
- Run on a scratch workspace, not your home directory: `--workspace /path/to/scratch`.
- On a shared cluster, do not set `--partition` to a queue where other users can read
  your job's working directory.
