# autoresearch

You are improving a single training script autonomously.

## Constraints
- Edit only the target code file.
- Do not add dependencies.
- Keep CLI runnable.
- Prefer simple changes with measurable gains.

## Loop
1. Read current code.
2. Propose one concrete improvement.
3. Implement it.
4. Run a smoke test.
5. Keep only if it improves metrics or stability.

## Metric Goal
- Minimize validation error/val metric.
- If no explicit metric is available, optimize for stable successful execution and cleaner outputs.

## Logging
- Write concise notes about what changed and why.
