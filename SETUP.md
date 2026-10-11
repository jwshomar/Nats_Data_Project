# Python Development Environment Setup

## Purpose
Reusable WSL2/Ubuntu environment for Bayesian statistics, scientific computing, and GPU-accelerated machine learning. Use VS Code connected to WSL2.

## Verified reference setup

| Component | Configuration |
| --- | --- |
| OS | Ubuntu 26.04.1 LTS under Windows WSL2 |
| Python | 3.13 (verified 3.13.16) |
| Package manager | uv |
| GPU | NVIDIA GeForce RTX 5080 (~16 GB) |
| Driver at verification | 591.86; driver-supported CUDA 13.1 |
| PyTorch | 2.14.1+cu130, CUDA 13.0 |
| JAX | 0.11.2, CUDA GPU backend |

## Files and dependency sources

- `pyproject.toml`: direct dependencies and package indexes. PyTorch uses `https://download.pytorch.org/whl/cu130`; JAX uses `jax[cuda13]`.
- `uv.lock`: resolved versions and sources; commit it to Git.
- `.python-version`: Python 3.13.
- `.venv/`: local environment; never copy or commit it.
- `AGENTS.md`: project-level Codex instructions.
- `requirements.txt`: optional legacy snapshot, **not** the source of truth.

## First-time prerequisites

Install WSL2, Ubuntu, VS Code with the WSL extension, and uv. In Ubuntu verify `nvidia-smi` can see the GPU. Do not install a Linux NVIDIA graphics driver in WSL2.

If uv is not installed, review its installer and run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.13
```

## Install or restore dependencies

From the project root:

```bash
uv sync --locked
```

This creates/synchronizes `.venv` from the lockfile. GPU packages may be large.

## Everyday development

In Windows PowerShell run `wsl`. In Ubuntu:

```bash
cd ~/projects/YOUR_PROJECT_NAME
code .
```

Select `./.venv/bin/python` in VS Code. No manual activation is needed. Run commands from the project root:

```bash
uv run --locked python your_script.py
uv run --locked pytest
uv run --locked jupyter lab
```

`uv run --locked` **may synchronize installed packages**. For inspection or execution without synchronization:

```bash
uv run --no-sync python your_script.py
.venv/bin/python --version
```

If a different project's environment is activated, run `deactivate` before using uv.

## Verify GPU execution

```bash
uv run --locked python - <<'PY'
import torch
import jax
import jax.numpy as jnp
print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('JAX:', jax.__version__, 'Devices:', jax.devices())
assert torch.cuda.is_available()
assert any(d.platform == 'gpu' for d in jax.devices())
assert torch.ones(10, device='cuda').sum().item() == 10
x = jnp.ones(10)
assert all(d.platform == 'gpu' for d in x.devices())
assert float(x.sum()) == 10
print('GPU computations passed')
PY
uv pip check --python .venv/bin/python
```

## Changing dependencies

Ask before installing, removing, or upgrading dependencies, particularly GPU/CUDA components. For approved changes, use `uv add PACKAGE`, review changes to `pyproject.toml` and `uv.lock`, then rerun GPU tests. Do not use `uv lock --upgrade` casually.

## Creating another project

```bash
bash ~/projects/create_project.sh New_Project_Name
```

The generator copies the master template, excludes virtual environments and private data, updates project metadata, refreshes the lockfile, installs packages, and verifies GPU computation. It refuses to overwrite existing destinations and leaves failed partial projects for inspection.

## Reproducibility and safety

Commit `pyproject.toml`, `uv.lock`, and `.python-version`; exclude `.venv/`, secrets, datasets, and generated artifacts. Use explicit random seeds and appropriate train/validation/test splits. A passing GPU test does not by itself establish statistical correctness.
