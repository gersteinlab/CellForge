# What this changes

<!-- One or two sentences. What is different after this PR that was not before? -->

Closes #

## Type

- [ ] Bug fix
- [ ] New dataset / benchmark
- [ ] New baseline
- [ ] New code-generation backend or retrieval provider
- [ ] Core framework change
- [ ] Documentation only

## How it was verified

<!--
Paste the command you ran and its result. For changes that affect generated
science (prompts, agent roles, retrieval, scoring), a before/after research plan
or a metric comparison is much more convincing than "tests pass".
-->

```
python -m pytest tests -v
```

## Checklist

- [ ] `python -m pytest tests` passes locally
- [ ] New behaviour has a test, or I explain below why it cannot be tested
- [ ] Docs updated if user-facing behaviour, CLI flags, or env vars changed
- [ ] No API keys, dataset paths from my machine, or agent transcripts committed
- [ ] `CHANGELOG.md` updated under "Unreleased" for user-visible changes

## Notes for reviewers

<!-- Anything you are unsure about, or want a second opinion on. -->
