# Removed OpenHands Backend

The original OpenHands integration was Docker/UI oriented and mostly contained
startup and network-diagnosis scripts. It did not cleanly implement the current
CellForge contract:

```text
research_plan -> data/codes/<dataset>/result.py
research_plan + task_id -> data/codes/<dataset>/result_<task_id>.py
```

It was therefore removed from the supported backend registry. The only
supported backend is `codex`.

Old configuration names (`openhands`, `openhands-legacy`, and
`legacy-openhands`) now fail during backend creation with a migration error.
The Python module and class name remain importable only as a deprecated shim;
constructing the class also raises the same error. They are intentionally not
returned by `list_backends()`.

A future OpenHands integration should not revive the placeholder
`generate_code()` method. It should implement the same isolated-workspace
coding-agent runner used for Codex, return a structured run result,
and pass the deterministic checks in `Code_Generation.verifier`.
