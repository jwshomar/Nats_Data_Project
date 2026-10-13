# Hitter Contact Ability

An analysis of the **Senior Analyst Data Problem**, estimating the probability of contact **given a swing**. The notebook explores the supplied pitch data, fits a logistic model with jointly estimated batter and pitcher effects, evaluates held-out performance and calibration, and provides configurable predictions with uncertainty intervals.

## Files

| File | Purpose |
| --- | --- |
| [Senior Analyst Data Problem.pdf](Senior%20Analyst%20Data%20Problem.pdf) | Problem statement and requested deliverables. |
| [notebook/data_problem.ipynb](notebook/data_problem.ipynb) | Main analysis and prediction lookups. The directory is `notebook/` (singular). |
| [src/contact_model.py](src/contact_model.py) | Data validation, splitting, fitting, predictions, and uncertainty calculations. |
| [tests/test_contact_model.py](tests/test_contact_model.py) | Numerical and behavioral checks for the model. |
| [data_problem_writeup.html](data_problem_writeup.html) | Self-contained report with an offline batter-selection dropdown. |
| [SETUP.md](SETUP.md) | Detailed environment instructions and reference GPU configuration. |

## Provide the data

The dataset is **not included in Git**. Obtain the supplied CSV separately and place it at this exact path, relative to the repository root:

```text
Nats_Data_Project/
├── data/
│   └── wsh_pitches.csv
├── notebook/
│   └── data_problem.ipynb
├── src/
└── README.md
```

Create `data/` if necessary:

```bash
mkdir -p data
```

Copy the dataset into that directory and name it **`wsh_pitches.csv`**. Keep the header row and these columns:

| Column | Meaning / expected values |
| --- | --- |
| `batter_id` | Stable numeric batter identifier. |
| `pitcher_id` | Stable numeric pitcher identifier. |
| `bat_side` | Batter's side: `L` or `R`. |
| `throw_side` | Pitcher's throwing side: `L` or `R`. |
| `is_fb` | `1` for a fastball; otherwise `0`. |
| `is_in_zone` | `1` for in-zone; otherwise `0`. |
| `is_swing` | `1` for a swing; otherwise `0`. |
| `is_miss` | `1` for a swing and miss; otherwise `0`. |

Each row represents one pitch. Required fields must be nonmissing, a miss must also be a swing, and each player ID must have one observed handedness. The current implementation does not handle switch-hitting. Alternative datasets also need enough swings in each context and for each player to support the split requirements; the notebook reports failures rather than silently dropping players.

## Set up and open the notebook

Use **Python 3.13**, **uv**, and a project-local `.venv`. The reference platform is Ubuntu under WSL2. From the repository root, install the locked environment:

```bash
uv sync --locked
```

`pyproject.toml` and `uv.lock` define the environment; `requirements.txt` is a legacy snapshot. The notebook's model runs on CPU. The broader locked environment also includes CUDA-enabled PyTorch and JAX packages, so installation can be substantial; see [SETUP.md](SETUP.md) for platform and GPU details.

Open the notebook with JupyterLab:

```bash
uv run --locked jupyter lab notebook/data_problem.ipynb
```

Alternatively, open the repository in VS Code, open `notebook/data_problem.ipynb`, and select **`.venv/bin/python`** as the notebook kernel. Ensure JupyterLab also uses the project environment.

Restart the kernel and **Run All** from top to bottom. This loads the CSV, establishes the split, selects shrinkage on validation data, fits the model, computes uncertainty, and produces the evaluation plots. Lookup cells depend on those earlier cells. Data loading supports a working directory of either the repository root or `notebook/`.

If the environment is already installed and you want execution without synchronization, use `uv run --no-sync` in place of `uv run --locked`.

## Use the prediction lookups

After running the model sections, edit the inputs in the lookup cell and rerun that cell.

### Lookup 1: the three requested scenarios

The default is batter 1, with batting side inferred from the data:

```python
batter_id = 1
bat_side = None
```

To predict for a neutral-quality LHH or RHH instead of a specific player:

```python
batter_id = None
bat_side = "L"  # Or "R".
```

Outputs cover all in-zone pitches, in-zone pitches against LHP, and in-zone fastballs against LHP. They use neutral pitcher quality and a common development-set pitch mixture within batting side. If an ID and batting side are both supplied, they must agree. Choose an ID present in your dataset.

### Lookup 2: a fully specified context

For example, a neutral LHH against neutral RHP on an out-of-zone fastball:

```python
context_batter_id = None
context_bat_side = "L"
context_throw_side = "R"
context_is_in_zone = 0
context_is_fb = 1
```

Use `L`/`R` for handedness, `1`/`0` for IZ/OOZ, and `1`/`0` for FB/non-FB. To assess a specific batter, supply `context_batter_id` and leave `context_bat_side = None` to infer their side. All 16 pitch contexts are supported.

Both lookups return contact probability conditional on a swing and approximate 95% probability intervals. “Neutral” refers to zero centered player effect on the logit scale, rather than a pooled observed contact rate.

## Outputs and interpretation

Running the notebook creates `outputs/contact_model/` with split assignments, selected settings, fitted coefficients and covariance, validation/test metrics, held-out predictions, reference weights, and simulation diagnostics. `data/`, `outputs/`, and `.venv/` are ignored by Git.

The notebook uses a reproducible 60/20/20 context-stratified split with every batter and pitcher represented in each partition. Selected models are refit on training plus validation; test outcomes stay outside fitting, tuning, and reference weights. Intervals condition on validation-selected shrinkage and the model assumptions. Without dates or game identifiers, evaluation concerns held-out swings involving known players, rather than validated future-season forecasts. The supplied outcomes are treated as ground truth, including their unusual platoon patterns.

Open `data_problem_writeup.html` directly in a browser to read the report and select any of the original 200 batters without running Python. Its figures and estimates are embedded and work offline. The report is a snapshot of the original analysis: rerunning the notebook does **not** regenerate the HTML or its dropdown values.

## Checks

From the repository root:

```bash
uv run --locked pytest -q
uv run --locked ruff check src/contact_model.py tests/test_contact_model.py notebook/data_problem.ipynb
```

The model tests use synthetic data and do not require `data/wsh_pitches.csv`. For a missing-file error in the notebook, check the CSV's exact filename and location. For import errors, check that the selected kernel is the project-local environment. After editing the helper module, restart the kernel and rerun the notebook to load the updated code.
