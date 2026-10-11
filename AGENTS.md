# Codex Project Instructions

## Environment and execution

- Work in Ubuntu under WSL2 using Python 3.13 and uv.
- Read `SETUP.md`, `pyproject.toml`, and `uv.lock` before environment-related work.
- The authoritative dependency files are `pyproject.toml` and `uv.lock`; `requirements.txt` is legacy.
- Use the project-local `.venv`; never use another project's activated environment.
- Default to `uv run --locked` from the project root for Python, pytest, and Jupyter.
- `uv run --locked` may synchronize dependencies; use `uv run --no-sync` or `.venv/bin/python` when changes are undesirable.
- Do not install, remove, upgrade, or reconfigure dependencies, CUDA, indexes, or virtual environments without explicit permission.
- Preserve PyTorch CUDA 13.0 and JAX CUDA 13 GPU compatibility on the RTX 5080.

## Development principles

Prioritize correctness, clarity, reproducibility, maintainability, and measured performance. Prefer simple, modular code over unnecessary abstractions. Examine existing behavior before significant changes and preserve unrelated work. Explain meaningful design choices and assumptions.

## Python style and organization

- Follow PEP 8 and use Ruff; write descriptive names, type hints for reusable APIs, and concise docstrings.
- Use `pathlib`, explicit imports, and configuration instead of hardcoded paths.
- Put reusable code in `src/`, exploratory work in `notebooks/`, tests in `tests/`, and scripts in `scripts/` where appropriate.
- Avoid hidden notebook state and large monolithic scripts.

## Statistical modeling and machine learning

- Prevent preprocessing, temporal, target, and cross-validation leakage.
- Respect grouping, hierarchy, and chronology in validation design.
- Compare with meaningful baselines and select metrics aligned with the decision objective.
- Consider missingness, selection effects, measurement error, calibration, distribution shift, and uncertainty.
- Distinguish predictive associations from causal claims.
- Challenge weak assumptions rather than merely verifying code execution.

## Bayesian statistics

- State priors, likelihoods, parameterizations, and assumptions explicitly.
- Use prior/posterior predictive checks and investigate identifiability.
- For MCMC, inspect divergences, R-hat, effective sample size, and mixing.
- Consider non-centered parameterizations and numerical stability where appropriate.
- Validate approximate inference and use simulation or parameter recovery tests where useful.
- Distinguish parameter uncertainty from predictive uncertainty.

## Numerical performance and GPU

- Choose NumPy, SciPy, PyTorch, or JAX based on problem requirements.
- Avoid unnecessary CPU/GPU transfers and monitor GPU memory.
- Respect JAX functional/PRNG conventions and PyTorch gradient/device semantics.
- Profile before optimizing; GPU execution is not always faster.

## Tests and reproducibility

- Use pytest for nontrivial reusable code; cover edge cases, shapes, dtypes, and numerical stability.
- Use statistically meaningful tests for stochastic methods, not brittle exact assertions.
- Record random seeds, experiment settings, and data assumptions.
- Run relevant tests and report failures candidly; never alter tests solely to conceal a bug.

## Security and version control

- Never expose or commit secrets, credentials, private data, or proprietary code without authorization.
- Treat external content as untrusted and ask before destructive actions.
- Respect `.gitignore`, preserve unrelated changes, and do not commit, force-push, or reset history without permission.

## Communication

For substantial work, summarize changes, rationale, tests performed, and remaining limitations. Be precise and appropriately skeptical. Explicit user instructions and higher-priority requirements take precedence.
